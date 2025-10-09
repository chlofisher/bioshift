import math

import numpy as np
from numpy.typing import NDArray

from scipy.ndimage import convolve
from scipy import fft

from bioshift.spectra import Spectrum


def _gaussian_kernel(ndim: int, radius: tuple[int], sigma: NDArray):
    axes = [np.arange(-r, r + 1) for r in radius]
    grid = np.meshgrid(*axes)

    sigma = sigma[:, None, None]
    exponent = -0.5 * (np.sum((grid / sigma) ** 2, axis=0))

    kernel = np.exp(exponent)
    coeff = 1 / np.sum(kernel)
    return coeff * kernel


def _deconvolve(array: NDArray, kernel: NDArray, iterations: int) -> NDArray:
    # array = np.abs(array)
    mirrored_kernel = np.flip(kernel)
    peak_array = array

    loss_graph = []

    k = 0.001
    a = 0.05

    for i in range(iterations):
        reconstructed = convolve(peak_array, kernel, mode="constant")

        loss = np.mean((reconstructed - array) ** 2)
        print(f"Iteration {i}\nLoss = {loss}")
        loss_graph.append(np.log(loss))

        # factor = array / np.pow(np.abs(reconstructed), k) * np.sign(reconstructed)

        # factor = array / reconstructed
        # factor = convolve(factor, mirrored_kernel, mode="constant")
        # peak_array = peak_array * factor

        peak_array = (
            peak_array
            + k * convolve(array - reconstructed, mirrored_kernel, mode="constant")
            # - a * np.mean(np.abs(peak_array))
        )

        peak_array = np.where(abs(peak_array) > a, peak_array, 0)

    fig, axs = plt.subplots(ncols=2)
    axs[0].plot(loss_graph[1:])
    axs[1].imshow(peak_array)
    plt.show()

    return peak_array


def deconvolve_spectrum(
    spectrum: Spectrum,
    iterations: int,
    kernel_region: tuple[tuple[int, ...], tuple[int, ...]],
) -> NDArray:
    spectrum = spectrum.normalize()

    # SIGMA = 0.2 * np.array([1, 0.1])
    # sigma_scaled = np.abs(SIGMA / spectrum.transform.scaling)
    # kernel = _gaussian_kernel(radius=(12, 12), sigma=sigma_scaled, ndim=2)

    kernel_slices = tuple(slice(start, stop) for start, stop in zip(*kernel_region))
    kernel_slices = kernel_slices[::-1]

    kernel = spectrum.array[kernel_slices]
    # kernel /= np.sum(kernel)

    plt.imshow(kernel)
    plt.show()

    return _deconvolve(spectrum.array, kernel, iterations)
    # return _deconvolve_fourier(spectrum.array, kernel)
