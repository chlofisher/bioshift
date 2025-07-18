from typing import Protocol
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter

from bioshift.core.spectrum import Spectrum


@dataclass(frozen=True)
class Peak:
    # Index is a float since the center of a peak
    # can be located between grid points
    coord: NDArray
    shift: NDArray


@dataclass(frozen=True)
class PeakList:
    peaks: NDArray
    spectrum: Spectrum

    def __init__(self, peaks: NDArray, spectrum: Spectrum):
        self.peaks = peaks
        self.spectrum = spectrum

    def __iter__(self):
        yield from self.peaks

    def __getitem__(self, index):
        return self.peaks[index]


class PeakPicker(Protocol):
    def pick_peaks(self, spectrum: Spectrum) -> PeakList:
        ...


class DifferenceOfGaussiansPicker(PeakPicker):

    def __init__(self, threshold, min_distance, sigma, k):
        self.threshold = threshold
        self.sigma = sigma
        self.k = k

    def pick_peaks(self, spectrum: Spectrum) -> PeakList:
        filter_1 = gaussian_filter(spectrum.data, sigma=self.sigma)
        filter_2 = gaussian_filter(spectrum.data, sigma=self.sigma * self.k)

        difference = filter_1 - filter_2

        peaks = peak_local_max(
            difference,
            min_distance=self.min_distance,
            threshold_rel=self.threshold)

        peaks = spectrum.coord_to_shift(peaks)

        peaklist = PeakList(peaks, spectrum)
        return peaklist


class LocalMaximaPicker(PeakPicker):

    def __init__(self, threshold, min_distance):
        self.threshold = threshold
        self.min_distance = min_distance

    def pick_peaks(self, spectrum: Spectrum) -> PeakList:
        peaks = peak_local_max(
            spectrum.data,
            min_distance=self.min_distance,
            threshold_rel=self.threshold)

        peaks = spectrum.coord_to_shift(peaks)

        peaklist = PeakList(peaks, spectrum)
        return peaklist
