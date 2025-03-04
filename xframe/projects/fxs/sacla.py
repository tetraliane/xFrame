from dataclasses import dataclass
import re
from typing import Tuple, Union

import h5py
import numpy as np
import numpy.typing as npt
from ruamel.yaml import YAML

from .correlate import DataReader


class SaclaDataReader(DataReader):
    """DataReader for the SACLA run data format.

    Data stored in the SACLA run data format needs a special preprocess because
    it has been converted from the number of photons to analog-to-digital unit
    (ADU) before saved in HDF5 files. For detail of this conversion, see section
    3 of http://xfel.riken.jp/users/mpccd_detector/instructions_ver1.12.pdf.

    The locations of data in HDF5 files must be given as a path-like string,
    e.g. `/path/to/data.h5/name/of/dataset`.
    """

    sacla_settings: "SaclaSettings"

    def __init__(self, *args, sacla_settings: str):
        super().__init__(*args)
        self.sacla_settings = SaclaSettings.load(sacla_settings)

    def _read_binary_2D_arr(
        self, fname: str, shape: Tuple[int, int], dtype="f", bo="<"
    ) -> npt.NDArray[np.float32]:
        data = read_dataset(fname, shape).astype(np.float32)

        if self.sacla_settings.background is not None:
            bg = read_dataset(self.sacla_settings.background, shape)
            data -= bg

        data *= (
            self.sacla_settings.detector_system_gain
            / self.sacla_settings.photon_energy
            * 3.65
        )

        return data


@dataclass
class SaclaSettings:
    detector_system_gain: float
    photon_energy: float
    background: Union[str, None] = None

    @classmethod
    def load(cls, path: str) -> "SaclaSettings":
        with open(path) as f:
            s = YAML(typ="safe").load(f)
        if not isinstance(s, dict):
            raise Exception("invalid settings")

        return cls(
            detector_system_gain=float(s["detector_system_gain"]),
            photon_energy=float(s["photon_energy"]),
            background=s.get("background", None),
        )


FNAME_RE = re.compile(r"(.+\.h5)(.+)")


def read_dataset(path: str, shape: Tuple[int, int]):
    m = FNAME_RE.match(path)
    if m is None:
        raise Exception(f"invalid path: {path}")
    file, name = m.group(1, 2)

    with h5py.File(file) as f:
        d = f[name][:]
    assert d.shape == shape
    return d
