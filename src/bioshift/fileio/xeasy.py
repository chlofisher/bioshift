from pathlib import Path
import re
import numpy as np

from bioshift.core.spectrumtransform import SpectrumTransform
from bioshift.fileio.spectrumreader import SpectrumReader


class XEASYSpectrumReader(SpectrumReader):
    params_path: Path
    data_path: Path

    def _get_params(self):
        with open(self.params_path, "r") as params_file:
            pattern = r"(^.+)( \.+ )(.+$)"
            text = params_file.read()

            matches = re.findall(pattern, text, flags=re.MULTILINE)

        params = {match.group(1): match.group(3) for match in matches}

        return params

    @classmethod
    def from_path(cls, path: Path):
        ...

    @classmethod
    def can_read(cls, path: Path) -> bool:
        return path.suffixes[0] == "3D"

    def get_data(self):
        ...

    def get_ndim(self):
        return int(self.params["Number of dimensions"])

    def get_nuclei(self):
        ...

    # TODO Figure out a way to get full referencing information for XEASY
    # spectra. Since the .3D.param file does not contain the reference
    # point or shift, the transform can not be determined solely from the 
    # spectrum. 
    def get_transform(self):
        shape = self._get_shape()

        spectrometer_frequency = ...

        transform = SpectrumTransform.from_reference(
            shape=shape,
            spectrometer_frequency=spectrometer_frequency,
        )

        return transform

    def _get_shape(self):
        shape = [None] * self.get_ndim()

        for key, val in self.params.items():
            val = int(val)
            pattern = r"^Size of spectrum in w(\d)$"
            key_match = re.match(pattern, key)

            if key_match:
                axis = int(key_match.group(1)) - 1
                shape[axis] = key

        return tuple(shape)

    def _get_spectrometer_frequency(self):
        frequency = [None] * self.get_ndim()

        for key, val in self.params.items():
            val = float(val)
            pattern = r"^Spectrometer frequency in w(\d)$"
            key_match = re.match(pattern, key)

            if key_match:
                axis = int(key_match.group(1)) - 1
                frequency[axis] = key

        return tuple(frequency)

    def _get_spectral_width(self):
        ...

    def _get_ref_coord(self):
        ...

    def _get_ref_shift(self):
        ...

