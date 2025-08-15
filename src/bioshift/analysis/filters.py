from typing import Callable
from numpy.typing import NDArray
import numpy as np
import skimage
import scipy
from functools import wraps, partial

from bioshift.analysis.pipeline import functionregistry
from bioshift.core.spectrum import Spectrum
from bioshift.fileio.spectrumdatasource import TransformedDataSource


# spectrum_filter converts functions on NDArrays to functions on Spectra.
# Output spectrum must have the same transform, shape, etc. as input.
# The transformation must be performed in array grid coordinates.
def spectrum_filter(func: Callable):
    @wraps(func)
    def wrapper(spectrum: Spectrum, **kwargs) -> Callable:
        new_data_source = TransformedDataSource(
            parent=spectrum.data_source, func=partial(func, **kwargs)
        )

        return Spectrum(
            ndim=spectrum.ndim,
            nuclei=spectrum.nuclei,
            data_source=new_data_source,
            transform=spectrum.transform,
        )

    return wrapper


# Can't use spectrum_filter decorator as the scaled sigma values depend on the
# spectrum transform.
@functionregistry.register()
def gaussian(input: Spectrum, sigma: tuple[float, ...]) -> Spectrum:
    if len(sigma) != input.ndim:
        raise ValueError(
            f"""Mismatched dimensions. Input spectrum is {input.ndim}D,
            but a {len(sigma)}D vector of sigmas was provided."""
        )

    sigma_scaled = np.array(sigma) / input.transform.scaling
    gaussian_filter = partial(skimage.filters.gaussian, sigma=sigma_scaled)

    # Manually create a spectrum_filter function using the gaussian blur with
    # the scaled sigmas and call it on the input.
    return spectrum_filter(func=gaussian_filter)(input)


@functionregistry.register()
@spectrum_filter
def laplacian(input: NDArray) -> NDArray:
    return scipy.ndimage.laplace(input)


@functionregistry.register()
@spectrum_filter
def threshold(input: NDArray, level: float) -> NDArray:
    return np.where(np.abs(input) < level, 0, input)


@functionregistry.register()
@spectrum_filter
def normalize(input: NDArray) -> NDArray:
    return input / np.abs(input).max()


@functionregistry.register()
@spectrum_filter
def abs(input: NDArray) -> NDArray:
    return np.abs(input)


@functionregistry.register()
@spectrum_filter
def positive(input: NDArray) -> NDArray:
    return np.where(input > 0, input, 0)


@functionregistry.register()
@spectrum_filter
def negative(input: NDArray) -> NDArray:
    return np.where(input < 0, input, 0)


@functionregistry.register()
def crop(input: Spectrum, bounds: list[tuple[float, float], ...]) -> Spectrum:
    if len(bounds != input.ndim):
        raise ValueError(
            f"""Mismatched dimensions. Input spectrum is
            {input.ndim}D, but a {len(bounds)}D list of bounds was provided."""
        )

    result = np.copy(input)

    slices = tuple(slice(start, end) for start, end in bounds)

    mask = np.zeros(input.shape, dtype=bool)
    mask[slices] = True

    result[~mask] = 0

    return result


@functionregistry.register()
def difference_of_gaussians(
    input: Spectrum, sigma: tuple[float, ...], k: float
) -> Spectrum:
    sigma_scaled = np.array(sigma) / input.transform.scaling

    print(sigma_scaled)

    gaussian1 = partial(skimage.filters.gaussian, sigma=sigma_scaled)
    gaussian2 = partial(skimage.filters.gaussian, sigma=k * sigma_scaled)

    scaling = 2 / (k * k - 1)

    return spectrum_filter(
        func=lambda arr: scaling * (gaussian1(arr) - gaussian2(arr))
    )(input)
