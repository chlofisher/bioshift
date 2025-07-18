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
    spectrum_shape: NDArray
    spectral_width: NDArray
    spectrometer_frequency: NDArray
    ref_coord: NDArray
    ref_ppm: NDArray

    def __post_init__(self):
        """Validation logic for the reference attributes.

        Raises:
            ValueError: If not all of the fields are 1D arrays.
            ValueError: If not all the fields have the same length.
            ValueError: If any of the values in spectrum_shape, spectral_width
                or spectrometer_frequency are non-positive.
        """
        arr_fields = ['spectrum_shape', 'spectral_width',
                      'spectrometer_frequency', 'ref_coord', 'ref_ppm']

        shapes = [getattr(self, name).shape for name in arr_fields]
        ndims = [getattr(self, name).ndim for name in arr_fields]

        if not all(n == 1 for n in ndims):
            raise ValueError(
                f"""All reference fields must be 1-dimensional, got shapes:
                 {shapes}""".strip()
            )

        if len(set(shapes)) != 1:
            raise ValueError(
                f"""All reference fields must have the same shape, got shapes:
                 {shapes}""".strip()
            )

        if np.any(self.spectrum_shape <= 0):
            raise ValueError("Shape values must be strictly positive")

        if np.any(self.spectral_width <= 0):
            raise ValueError("Spectral width values must be strictly positive")

        if np.any(self.spectrometer_frequency <= 0):
            raise ValueError("Spectrometer freq. must be strictly positive")

    def transform(self) -> SpectrumTransform:
        """Transform object derived from the referencing information. 
        Uses linear relationship between spectrum coordinates and chemical 
        shifts to determine the scaling and offset of the transformation.

        Returns:
            SpectrumTransform object corresponding to provided spectrum 
            referencing.
        """
        w = self.spectral_width
        N = self.spectrum_shape
        f = self.spectrometer_frequency

        scaling = w / (N * f)
        offset = self.ref_ppm - scaling * self.ref_coord

        return SpectrumTransform(scaling, offset)
