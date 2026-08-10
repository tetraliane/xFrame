import os
import logging
from dataclasses import dataclass
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from xframe.database.database import SettingsParser
from xframe.interfaces import ProjectWorkerInterface
from xframe.settings.tools import DictNamespace
from ._database_ import ProjectDB
from .average import Alignment
from .projectLibrary.classes import FTGridPair
from .projectLibrary.misk import get_analysis_process_factory
from .projectLibrary.resolution_metrics import smooth_gaussian
from .prtf import ComplexFunc, generate_ft

logger = logging.getLogger(__name__)


class ProjectWorker(ProjectWorkerInterface):
    settings: DictNamespace
    db: ProjectDB
    ft: ComplexFunc

    def run(self):
        average_settings = self.db.load("average_settings", path_modifiers={})
        SettingsParser(None).recursive_command_execution(average_settings, None, None)
        reconst_settings = self.db.load("reconstruction_settings", path_modifiers={})

        recip_coef = reconst_settings["fourier_transform"]["reciprocity_coefficient"]
        grid = self.load_grid()
        reconst_settings["internal_grid"] = FTGridPair(*grid)

        ft, _ = generate_ft(
            grid.real,
            mode=self.settings.get("fourier_transform", {}).get("mode", "midpoint"),
            max_order=self.settings.get("fourier_transform", {}).get("max_order", 30),
            reciprocity_coef=recip_coef,
        )
        self.ft = ft

        data = self.load_data(grid.real.shape[:3])

        aligner = create_aligner(average_settings, reconst_settings, self.db, grid.real)
        average1 = DataPair(*aligner.shift_to_center(*data.average1)[:2])
        average2 = DataPair(*aligner.shift_to_center(*data.average2)[:2])
        aligned = aligner.apply_to(average1.real, average2)
        new_average2 = aligned["densities"]

        q = grid.reciprocal[:, 0, 0, 0]
        fsc = calc_fsc(ft, average1.real, new_average2[0])
        smoothed = smooth(q, fsc, self.settings.get("smoothing"))

        if "fsc" in self.db.files:
            self.db.save(
                "fsc",
                {"q": q, "fsc": fsc, "smooth_fsc": smoothed},
                skip_custom_methods=False,
                path_modifiers={"name": self.settings["name"]},
            )
        if "fsc_plot" in self.db.files:
            self.db.save(
                "fsc_plot",
                plot(q, fsc),
                skip_custom_methods=False,
                path_modifiers={"name": self.settings["name"]},
            )
        if "smooth_fsc_plot" in self.db.files:
            for mode, y in smoothed.items():
                self.db.save(
                    "smooth_fsc_plot",
                    plot(q, y),
                    skip_custom_methods=False,
                    path_modifiers={
                        "name": self.settings["name"],
                        "smoothing_mode": mode,
                    },
                )
        if "aligned_averages_vtk" in self.db.files:
            self.db.save(
                "aligned_averages_vtk",
                [np.real(average1.real), np.real(new_average2[0])],
                dset_names=["average1", "average2"],
                grid=grid.real,
                grid_type="spherical",
                skip_custom_methods=True,
                path_modifiers={"name": self.settings["name"]},
            )

    def load_data(self, shape: tuple[int, int, int]) -> "LoadedData":
        if "1_average" in self.db.files:
            data = self.load_from_average("1_average")
        elif "1_binary" in self.db.files:
            data = self.load_from_binary("1_binary", shape)
        else:
            raise ValueError("Required data not found in the database.")

        if "2_average" in self.db.files:
            data2 = self.load_from_average("2_average")
        elif "2_binary" in self.db.files:
            data2 = self.load_from_binary("2_binary", shape)
        else:
            raise ValueError("Required data not found in the database.")

        return LoadedData(
            average1=data,
            average2=data2,
        )

    def load_from_average(self, filename: str) -> "DataPair":
        with self.db.load(filename, as_h5_object=True) as f:
            average_reconst1 = DataPair(
                real=f["centered_average/real_density"][()],
                reciprocal=f["centered_average/reciprocal_density"][()],
            )

        return average_reconst1

    def load_from_binary(
        self, filename: str, shape: tuple[int, int, int]
    ) -> "DataPair":
        path = os.path.join(
            self.db.folders[self.db.files[filename]["folder"]],
            self.db.files[filename]["name"],
        )
        dtype = self.db.files[filename]["options"]["dtype"]
        data = np.fromfile(path, dtype=dtype).reshape(shape).astype(np.complex128)
        data_ft = self.ft(data)
        return DataPair(data, data_ft)

    def load_grid(self) -> "DataPair":
        filename = "grid_info" if "grid_info" in self.db.files else "1_average"
        with self.db.load(filename, as_h5_object=True) as f:
            grid = DataPair(
                real=f["internal_grid/real_grid"][()],
                reciprocal=f["internal_grid/reciprocal_grid"][()],
            )
        return grid


@dataclass
class LoadedData:
    average1: "DataPair"
    average2: "DataPair"


class DataPair(NamedTuple):
    real: npt.NDArray
    reciprocal: npt.NDArray


def create_aligner(
    average_settings: DictNamespace,
    reconst_settings: DictNamespace,
    db: ProjectDB,
    grid: npt.NDArray,
) -> Alignment:
    n_r, n_theta, n_phi = grid.shape[:3]
    aligner_options = {
        "opt": {
            "find_rotation": average_settings["find_rotation"],
            "max_iterations": average_settings["max_iterations"],
            "alignment_error_limit": average_settings["alignment_error_limit"],
        },
        "r_opt": {
            "grid": {
                "max_order": reconst_settings["grid"]["max_order"],
                "n_radial_points": n_r,
                "n_theta": n_theta,
                "n_phi": n_phi,
            },
            "GPU": reconst_settings["GPU"],
            "internal_grid": reconst_settings["internal_grid"],
            "fourier_transform": reconst_settings["fourier_transform"],
        },
    }

    return Alignment(
        get_analysis_process_factory(None, None),
        db,
        aligner_options,
    )


def calc_fsc(
    ft: ComplexFunc,
    reconst1: npt.NDArray[np.float64],  # shape (Nr, Ntheta, Nphi)
    reconst2: npt.NDArray[np.float64],  # shape (Nr, Ntheta, Nphi)
) -> npt.NDArray[np.float64]:
    reconst1_ft = ft(reconst1)
    reconst2_ft = ft(reconst2)

    fsc = np.zeros(reconst1_ft.shape[0])
    for i in range(reconst1_ft.shape[0]):
        num = np.sum(np.real(reconst1_ft[i] * np.conj(reconst2_ft[i])))
        denom = np.sqrt(
            np.sum(np.abs(reconst1_ft[i]) ** 2) * np.sum(np.abs(reconst2_ft[i]) ** 2)
        )
        fsc[i] = num / denom if denom > 0 else 0

    return fsc


def smooth(
    q: npt.NDArray[np.float64],
    fsc: npt.NDArray[np.float64],
    settings: dict | None,
) -> dict[str, npt.NDArray[np.float64]]:
    settings = settings or {}
    smoothed = dict()

    use_gaussian = settings.get("gaussian", {}).get("use", False)
    if use_gaussian:
        logger.info("Applying Gaussian smoothing to FSC curve")

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

        smoothed["gaussian"] = smooth_gaussian(q, fsc, sigma)

    return smoothed


def plot(
    q: npt.NDArray[np.float64],
    fsc: npt.NDArray[np.float64],
) -> plt.Figure:
    fig = plt.figure(layout="constrained")
    ax = fig.add_subplot()
    ax.plot(q, fsc)
    ax.axhline(0.5, color="black", linestyle="--")
    ax.set_xlabel("$q$ / $\\mathrm{\\AA}^{-1}$")
    ax.set_ylabel("FSC")
    ax.set_ylim(-0.1, 1.1)
    ax.grid()
    return fig
