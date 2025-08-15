import numpy as np
from numpy.typing import NDArray
import math

from bioshift.core.spectrumdatasource import SpectrumDataSource
from bioshift.core.spectrumreference import SpectrumReference


class MockDataSource(SpectrumDataSource):
    peaklist: list[NDArray]

    def __init__(self, peaks: list[NDArray]):
        self.peaks = peaks
        self.cache = None

    def _load_data(self) -> NDArray:
        spectrum = np.zeros(self.params.shape)

        for peak in self.peaks:
            peak_contribution = np.fromfunction(
                function=lambda *args: self.lineshape(peak, 9, *args),
                shape=self.params.shape,
            )

            spectrum += peak_contribution

        noise = self.noise(self.params.shape)
        spectrum += noise * 0.05

        return spectrum

    @classmethod
    def random_spectrum(cls, shape, n_peaks):
        peaks = np.random.rand(n_peaks, len(shape)) * shape

        reference = SpectrumReference(
            spectrum_shape=np.array(shape),
            spectral_width=np.array([1, 1]),
            spectrometer_frequency=np.array([1, 1]),
            ref_coord=np.array((0, 0)),
            ref_ppm=np.array((1, 1)),
        )

        params = SpectrumParams(
            ndim=len(shape),
            shape=shape,
            block_shape=None,
            n_blocks=None,
            nuclei=None,
            transform=reference.transform,
        )

        return cls(params=params, peaks=peaks)

    def noise(self, shape):
        rng = np.random.default_rng()

        noise = rng.normal(0, 1, size=math.prod(shape))
        noise = noise.reshape(shape)

        return noise

    def lineshape(self, center, gamma, *args):
        # return self.gaussian_lineshape(center, gamma, *args)
        return self.lorentzian_lineshape(center, gamma, *args)

    def gaussian_lineshape(self, center, gamma, *args):
        intensity = 1

        for x, x0 in zip(args, center):
            intensity *= np.exp(-np.log(2) * np.power((x0 - x) / gamma, 2))

        return intensity

    def lorentzian_lineshape(self, center, gamma, *args):
        intensity = 1

        for x, x0 in zip(args, center):
            intensity += np.power((x - x0) / gamma, 2)

        intensity = 1 / (1 + intensity)

        return intensity
