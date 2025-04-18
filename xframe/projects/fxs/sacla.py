from dataclasses import dataclass
import os
import re
import sys
import time
from typing import Tuple, Union

import h5py
import numpy as np
import numpy.typing as npt
from ruamel.yaml import YAML

from xframe import database, settings
from .correlate import DataReader
from .projectLibrary.cross_correlation import ccf_analysis


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

    def __init__(
        self,
        compute,
        patterns_max,
        batch_size,
        numcpus_max,
        list_inp,
        image_dimensions,
        intensity_pixel_threshold,
        intensity_radial_pixel_filter,
        mask_binary,
        background_subtraction,
        ROInormalization,
        ROImeanfilter,
        pixsz,
        det_sam,
        wavelng,
        xpolarization,
        solid_angle_correction,
        qrange,
        qrange_xcca,
        fc_n_max,
        phirange,
        ccf_2p_symmetrize,
        dpcenter,
        interp_order,
        sacla_settings: str,
    ):
        self.start_time = time.time()
        self.compute = compute
        self.patterns_max = patterns_max
        self.batch_size = batch_size
        self.numcpus_max = numcpus_max
        self.split_mode = settings.project.split_mode
        self.list_inp = list_inp
        self.intensity_pixel_threshold = intensity_pixel_threshold
        self.intensity_radial_pixel_filter = intensity_radial_pixel_filter
        self.mask_binary_inp = mask_binary
        self.background_subtraction = background_subtraction
        self.ROInormalization = ROInormalization
        self.ROImeanfilter = ROImeanfilter
        self.interp_order = interp_order
        self.fc_n_max = fc_n_max
        self.ccf_2p_symmetrize = ccf_2p_symmetrize
        self.pixelsize = pixsz
        self.det_sam = det_sam
        self.wavelng = wavelng
        self.dpcenter = dpcenter
        self.xpolarization = xpolarization
        self.solid_angle_correction = solid_angle_correction
        self.img_shape = image_dimensions
        self.sacla_settings = SaclaSettings.load(sacla_settings)

        ### Below are copy of DataReader.__init__(), without the check of input file existence

        # analyze dependencies of computations and update the "compute" list
        self._analyse_dependencies()

        # if os.path.exists(self.dir_save) is not True:
        #        os.makedirs(self.dir_save)

        # print('\nResults output directory: {}'.format(self.dir_save))
        print("Input file with a file list: {}".format(self.list_inp))

        self.file_list = self._read_line_text(self.list_inp)
        self.M = len(self.file_list)

        self.good_frames = np.ones(
            self.M, dtype=int
        )  # array with labels for each frame "1"-good, "0"-bad; initially we assume that all frames are good

        # read binary mask that will be applied to all images
        if self.mask_binary_inp == True:
            path = database.project.get_path("binary_mask")
            if os.path.exists(path):
                self.mask_binary = self._read_binary_2D_arr(path, self.img_shape)
            else:
                print(
                    "Error: Input file {} with binary mask have not been found.\n".format(
                        path
                    )
                )
                sys.exit(1)

        # read background data from the file
        if self.background_subtraction:
            path = database.project.get_path("background")
            if os.path.exists(path):
                self.background_data = self._read_binary_2D_arr(path, self.img_shape)
            else:
                print(
                    "Error: Input file {} with background data have not been found.\n".format(
                        path
                    )
                )
                sys.exit(1)

        # determime the reciprocal space geometry and related exp qunatities
        self._prepare_polar_representation(qrange, qrange_xcca, phirange)

        # determine polarization and solid angle correction factors
        if self.xpolarization[0]:
            self._determine_polarization_correction()
        if self.solid_angle_correction == True:
            self._determine_solid_angle_correction()

        # define ROI for normalizing data
        if self.ROInormalization[0] or self.ROImeanfilter[0]:
            self.ROInorm_qpos1 = np.abs(self.qvals - self.ROInormalization[1]).argmin()
            self.ROInorm_qpos2 = np.abs(self.qvals - self.ROInormalization[2]).argmin()
            if self.ROInorm_qpos1 == self.ROInorm_qpos2:
                print(
                    "WARNING: ROI normalization range ({},{}) contains only 1 radial point".format(
                        self.qvals[self.ROInorm_qpos1], self.qvals[self.ROInorm_qpos2]
                    )
                )
            else:
                print(
                    "ROI normalization range ({},{}) contains {} radial points".format(
                        self.qvals[self.ROInorm_qpos1],
                        self.qvals[self.ROInorm_qpos2],
                        self.ROInorm_qpos2 - self.ROInorm_qpos1 + 1,
                    )
                )

        # initialize xcca functionality
        if "xcca" in self.compute:
            self.xcca_data = ccf_analysis(
                self.n_q1, self.n_q2, self.n_phi, self.q1vals_pos, self.q2vals_pos
            )

    def _read_binary_2D_arr(
        self, fname: str, shape: Tuple[int, int], dtype="f", bo="<"
    ) -> npt.NDArray[np.float32]:
        shape = tuple(shape)
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
