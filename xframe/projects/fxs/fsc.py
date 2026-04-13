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
from .prtf import ComplexFunc, generate_ft


class ProjectWorker(ProjectWorkerInterface):
    settings: DictNamespace
    db: ProjectDB

    def run(self):
        data = self.load_from_average()
        average_settings = self.db.load("average_settings", path_modifiers={})
        SettingsParser(None).recursive_command_execution(average_settings, None, None)
        reconst_settings = self.db.load("reconstruction_settings", path_modifiers={})
        reconst_settings["internal_grid"] = FTGridPair(*data.grid)

        aligner = create_aligner(average_settings, reconst_settings, self.db)
        aligned = aligner.apply_to(data.average1.real, data.average2)
        new_average2 = aligned["densities"]

        ft, _ = generate_ft(
            data.grid.real,
            mode=self.settings.get("fourier_transform", {}).get("mode", "midpoint"),
            max_order=self.settings.get("fourier_transform", {}).get("max_order", 30),
            reciprocity_coef=data.reciprocity_coef,
        )
        fsc = calc_fsc(ft, data.average1.real, new_average2[0])

        if "fsc" in self.db.files:
            self.db.save(
                "fsc",
                {"q": data.grid.reciprocal[:, 0, 0, 0], "fsc": fsc},
                skip_custom_methods=False,
                path_modifiers={"name": self.settings["name"]},
            )
        if "fsc_plot" in self.db.files:
            self.db.save(
                "fsc_plot",
                plot(data.grid.reciprocal[:, 0, 0, 0], fsc),
                skip_custom_methods=False,
                path_modifiers={"name": self.settings["name"]},
            )
        if "aligned_averages_vtk" in self.db.files:
            self.db.save(
                "aligned_averages_vtk",
                [np.real(data.average1.real), np.real(new_average2[0])],
                dset_names=["average1", "average2"],
                grid=data.grid.real,
                grid_type="spherical",
                skip_custom_methods=True,
                path_modifiers={"name": self.settings["name"]},
            )

    def load_from_average(self) -> "LoadedData":
        with self.db.load("average_result1", as_h5_object=True) as f:
            average_reconst1 = DataPair(
                real=f["centered_average/real_density"][()],
                reciprocal=f["centered_average/reciprocal_density"][()],
            )
            grid = DataPair(
                real=f["internal_grid"]["real_grid"][()],
                reciprocal=f["internal_grid"]["reciprocal_grid"][()],
            )
            reciprocity_coef = f["reciprocity_coefficient"][()]

        with self.db.load("average_result2", as_h5_object=True) as f:
            average_reconst2 = DataPair(
                real=f["centered_average/real_density"][()],
                reciprocal=f["centered_average/reciprocal_density"][()],
            )

        return LoadedData(
            average1=average_reconst1,
            average2=average_reconst2,
            grid=grid,
            reciprocity_coef=reciprocity_coef,
        )


@dataclass
class LoadedData:
    average1: "DataPair"
    average2: "DataPair"
    grid: "DataPair"
    reciprocity_coef: float


class DataPair(NamedTuple):
    real: npt.NDArray
    reciprocal: npt.NDArray


def create_aligner(
    average_settings: DictNamespace, reconst_settings: DictNamespace, db: ProjectDB
) -> Alignment:
    aligner_options = {
        "opt": {
            "find_rotation": average_settings["find_rotation"],
            "max_iterations": average_settings["max_iterations"],
            "alignment_error_limit": average_settings["alignment_error_limit"],
        },
        "r_opt": {
            "grid": {
                "max_order": reconst_settings["grid"]["max_order"],
                "n_radial_points": 128,
                "n_theta": 72,
                "n_phi": 140,
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
    ax.grid()
    return fig
