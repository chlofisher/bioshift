from typing import Callable
from numpy.typing import NDArray
import numpy as np
from skimage.filters import gaussian
from functools import wraps, partial

from bioshift.core.spectrum import Spectrum
from bioshift.analysis.analysisnode import analysis_node
from bioshift.fileio.spectrumdatasource import TransformedDataSource


def spectrum_filter(func: Callable):
    @wraps(func)
    def wrapper(spectrum: Spectrum, **kwargs) -> Callable:
        new_data_source = TransformedDataSource(
            parent=spectrum.data_source,
            func=partial(func, **kwargs)
        )

        return Spectrum(
            ndim=spectrum.ndim,
            nuclei=spectrum.nuclei,
            data_source=new_data_source,
            transform=spectrum.transform
        )

    return wrapper


@analysis_node
@spectrum_filter
def gaussian_filter(input: NDArray, sigma: float = 1.0) -> NDArray:
    return gaussian(input, sigma=sigma)


@analysis_node
@spectrum_filter
def threshold(input: NDArray, value=10.0):
    return np.where(np.abs(input) < value, 0, input)


@analysis_node
@spectrum_filter
def normalise(input: NDArray):
    return input / np.abs(input).max()


@analysis_node
@spectrum_filter
def difference_of_gaussians(input: NDArray,
                            sigma: float = 2.0, k: float = 1.5) -> NDArray:
    return gaussian(input, sigma=sigma) - gaussian(input, sigma=k*sigma)
