"""Implementation of the PRTF calculation using the Kurta method. 3D only."""

import logging
from dataclasses import dataclass
from typing import NamedTuple, Callable

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from xframe.interfaces import ProjectWorkerInterface
from xframe.settings.tools import DictNamespace
from ._database_ import ProjectDB
from .projectLibrary import fourier_transforms, hankel_transforms
from .projectLibrary.harmonic_transforms import HarmonicTransform
from .projectLibrary.resolution_metrics import smooth_gaussian

DIMENSIONS = 3
logger = logging.getLogger(__name__)


class ProjectWorker(ProjectWorkerInterface):
    settings: DictNamespace
    db: ProjectDB

    def run(self):
        data = self.load_from_average()

        ft, _ = generate_ft(
            data.grid.real,
            mode=self.settings.get("fourier_transform", {}).get("mode", "midpoint"),
            max_order=self.settings.get("fourier_transform", {}).get("max_order", 30),
            reciprocity_coef=data.reciprocity_coef,
        )

        q = data.grid.reciprocal[:, 0, 0, 0]
        prtf = prtf_kurta(ft, data.reconsts)
        smoothed = smooth(q, prtf, self.settings.get("smoothing"))

        if "prtf" in self.db.files:
            self.db.save(
                "prtf",
                {"q": q, "prtf": prtf, "smooth_prtf": smoothed},
                skip_custom_methods=False,
                path_modifiers={"name": self.settings["name"]},
            )
        if "prtf_plot" in self.db.files:
            fig = plot(q, prtf)
            self.db.save(
                "prtf_plot",
                fig,
                skip_custom_methods=False,
                path_modifiers={"name": self.settings["name"]},
            )
            plt.close(fig)
        if "smooth_prtf_plot" in self.db.files:
            for mode, y in smoothed.items():
                fig = plot(q, y)
                self.db.save(
                    "smooth_prtf_plot",
                    fig,
                    skip_custom_methods=False,
                    path_modifiers={
                        "name": self.settings["name"],
                        "smoothing_mode": mode,
                    },
                )
                plt.close(fig)

    def load_from_average(self) -> "LoadedData":
        with self.db.load("average_result", as_h5_object=True) as f:
            reconsts = []
            for key in f["aligned"].keys():
                real = f["aligned"][key]["real_density"][()]
                reciprocal = f["aligned"][key]["reciprocal_density"][()]
                reconsts.append(DataPair(real, reciprocal))

            grid = DataPair(
                real=f["internal_grid"]["real_grid"][()],
                reciprocal=f["internal_grid"]["reciprocal_grid"][()],
            )

            reciprocity_coef = f["reciprocity_coefficient"][()]

        return LoadedData(
            reconsts=reconsts, grid=grid, reciprocity_coef=reciprocity_coef
        )


@dataclass
class LoadedData:
    reconsts: list["DataPair"]
    grid: "DataPair"
    reciprocity_coef: float


class DataPair(NamedTuple):
    # `real` and `reciprocal` are 3D arrays of the same shape (Nr, Ntheta, Nphi).
    real: npt.NDArray[np.float64]
    reciprocal: npt.NDArray[np.float64]


ComplexFunc = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]


def generate_ft(
    grid_real: npt.NDArray[np.float64],
    mode="midpoint",
    max_order=30,
    reciprocity_coef=np.pi,
    use_gpu=False,
) -> tuple[ComplexFunc, ComplexFunc]:
    r_max = grid_real[:, 0, 0, 0].max()
    n_r = grid_real.shape[0]
    n_theta = grid_real.shape[1]
    n_phi = grid_real.shape[2]

    weights = hankel_transforms.generate_weightDict(
        max_order,
        n_r,
        dimensions=DIMENSIONS,
        mode=mode,
        reciprocity_coefficient=reciprocity_coef,
    )
    ht = HarmonicTransform(
        "complex",
        {
            "dimensions": DIMENSIONS,
            "max_order": max_order,
            "n_theta": n_theta,
            "n_phi": n_phi,
            # "anti_aliazing_degree": 2,
            # "indices": "lm",
        },
    )
    orders = np.arange(max_order + 1)

    ft, ift = fourier_transforms.generate_ft(
        r_max,
        weights,
        ht,
        DIMENSIONS,
        mode=mode,
        pos_orders=orders,
        reciprocity_coefficient=reciprocity_coef,
        use_gpu=use_gpu,
    )
    return ft, ift


def prtf_kurta(ft: ComplexFunc, reconsts: list[DataPair]) -> npt.NDArray[np.float64]:
    rho_ft_avg = np.mean(
        [ft(r.real) for r in reconsts],
        axis=0,
    )
    intensity_avg = np.mean(
        [(r.reciprocal * np.conj(r.reciprocal)).real for r in reconsts],
        axis=0,
    )
    prtf = np.abs(rho_ft_avg) / np.sqrt(intensity_avg)
    prtf_avg = np.mean(prtf, axis=(1, 2))
    return prtf_avg


def smooth(
    q: npt.NDArray[np.float64],
    prtf: npt.NDArray[np.float64],
    settings: dict | None,
) -> dict[str, npt.NDArray[np.float64]]:
    settings = settings or {}
    smoothed = dict()

    use_gaussian = settings.get("gaussian", {}).get("use", False)
    if use_gaussian:
        logger.info("Applying Gaussian smoothing to PRTF curve")

        sigma = settings.get("gaussian", {}).get("sigma")
        default_sigma = np.max(q) / 10

        if sigma is None:
            sigma = default_sigma
            logger.warning(
                f"Gaussian smoothing sigma not specified. Setting to: {sigma:.3f}"
            )

        try:
            sigma = float(sigma)
        except (ValueError, TypeError):
            sigma = default_sigma
            logger.warning(
                f"Invalid sigma value for Gaussian smoothing: {sigma}. Setting to: {sigma:.3f}"
            )

        smoothed["gaussian"] = smooth_gaussian(q, prtf, sigma)

    return smoothed


def plot(q: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> plt.Figure:
    fig = plt.figure(layout="constrained")
    ax = fig.add_subplot()
    ax.plot(q, y)
    ax.axhline(1 / np.e, color="black", linestyle="--")
    ax.set_xlabel("$q$ / $\\mathrm{\\AA}^{-1}$")
    ax.set_ylim(-0.1, 1.1)
    ax.set_ylabel("PRTF")
    ax.grid()
    return fig
