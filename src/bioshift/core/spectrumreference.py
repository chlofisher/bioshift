from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from bioshift.core.spectrumtransform import SpectrumTransform


@dataclass(frozen=True)
class SpectrumReference:
    """Contains the referencing information required to create the spectrum
    transform.

    Attributes:
        spectrum_shape: Number of data points along each axis of the spectrum.
        spectral_width: Width of the spectrum along each axis in Hz.
        spectrometer_frequency: Frequency of the spectrometer along each axis 
            in MHz.
        ref_coord: Array index coordinate vector of the reference point.
        ref_ppm: Chemical shift vector of the reference point.
    """
    spectrum_shape: tuple[int]
    spectral_width: tuple[float]
    spectrometer_frequency: tuple[float]
    ref_coord: tuple[float]
    ref_ppm: tuple[float]

    def transform(self) -> SpectrumTransform:
        """Transform object derived from the referencing information. 
        Uses linear relationship between spectrum coordinates and chemical 
        shifts to determine the scaling and offset of the transformation.

        Returns:
            SpectrumTransform object corresponding to provided spectrum 
            referencing.
        """
        w = np.array(self.spectral_width)
        N = np.array(self.spectrum_shape)
        f = np.array(self.spectrometer_frequency)

        delta_0 = np.array(self.ref_ppm)
        i_0 = np.array(self.ref_coord)

        scaling = w / (N * f)
        offset = delta_0 - scaling * i_0

        return SpectrumTransform(scaling, offset)
