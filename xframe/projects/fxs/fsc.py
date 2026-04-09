from dataclasses import dataclass
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from xframe.interfaces import ProjectWorkerInterface
from xframe.settings.tools import DictNamespace
from ._database_ import ProjectDB
from .prtf import ComplexFunc, generate_ft


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
        fsc = calc_fsc(ft, data.average1, data.average2)

        if "fsc" in self.db.files:
            self.db.save(
                "fsc",
                {"q": data.grid.reciprocal[:, 0, 0, 0], "fsc": fsc},
                skip_custom_methods=False,
                path_modifiers={"name": self.settings["name"]},
            )
        if "fsc_plot" in self.db.files:
            fig = plt.figure(layout="constrained")
            ax = fig.add_subplot()
            q = data.grid.reciprocal[:, 0, 0, 0]
            ax.plot(q, fsc)
            ax.axhline(0.5, color="black", linestyle="--")
            ax.set_xlabel("$q$ / $\\mathrm{\\AA}^{-1}$")
            ax.grid()
            self.db.save(
                "fsc_plot",
                fig,
                skip_custom_methods=False,
                path_modifiers={"name": self.settings["name"]},
            )

    def load_from_average(self) -> "LoadedData":
        with self.db.load("average_result1", as_h5_object=True) as f:
            average_reconst1 = f["average/real_density"][()]
            grid = DataPair(
                real=f["internal_grid"]["real_grid"][()],
                reciprocal=f["internal_grid"]["reciprocal_grid"][()],
            )
            reciprocity_coef = f["reciprocity_coefficient"][()]

        with self.db.load("average_result2", as_h5_object=True) as f:
            average_reconst2 = f["average/real_density"][()]

        return LoadedData(
            average1=average_reconst1,
            average2=average_reconst2,
            grid=grid,
            reciprocity_coef=reciprocity_coef,
        )


@dataclass
class LoadedData:
    average1: npt.NDArray[np.float64]
    average2: npt.NDArray[np.float64]
    grid: "DataPair"
    reciprocity_coef: float


class DataPair(NamedTuple):
    real: npt.NDArray[np.float64]
    reciprocal: npt.NDArray[np.float64]


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
