from numpy.typing import NDArray
# import numpy as np
import time
from jax import numpy as np
from jax import scipy, jit
from functools import partial

from matplotlib import pyplot as plt

from bioshift.core.spectrum import Spectrum


@partial(jit, static_argnums=3)
def _gaussian(x, mu, sigma, ndim=2):
    mu = np.array(mu)
    sigma = np.array(sigma)

    exponent = -0.5 * np.sum(((x - mu) / sigma) ** 2, axis=-1)
    coeff = 1 / (np.sqrt((2 * np.pi) ** ndim * np.linalg.norm(sigma)))

    return coeff * np.exp(exponent)


def restricted_pick(
    spectrum: Spectrum,
    peaks: NDArray,
    widths: NDArray,
    axis: int,
    peak_block_size: int = 6,
):
    slice = spectrum.slice(axis, z=0)

    coord_axes = [
        np.linspace(min, max, s)
        for s, min, max in zip(
            slice.shape, slice.transform.bounds[0], slice.transform.bounds[1]
        )
    ]

    mesh = np.meshgrid(*coord_axes, indexing="ij")

    coords = np.array(mesh)
    coords = np.repeat(coords[:, :, :, None], peaks.shape[0], axis=3)
    coords = np.moveaxis(coords, 0, -1)

    n_peaks = peaks.shape[0]
    n_blocks = (n_peaks // peak_block_size) + 1

    peak_blocks = np.array_split(peaks, n_blocks, axis=0)
    width_blocks = np.array_split(widths, n_blocks, axis=0)
    coord_blocks = np.array_split(coords, n_blocks, axis=2)

    results = []
    for coord_block, peak_block, width_block in zip(
        coord_blocks, peak_blocks, width_blocks
    ):
        gaussians = _gaussian(
            x=coord_block, mu=peak_block, sigma=np.array((0.25, 0.05)), ndim=spectrum.ndim - 1
        )

        gaussians = np.expand_dims(gaussians, axis=axis)
        filtered = gaussians * np.expand_dims(spectrum.array, axis=-1)

        integral_axes = [ax for ax in range(spectrum.ndim+1) if ax not in [axis, spectrum.ndim]]
        res = filtered

        for ax in sorted(integral_axes, reverse=True):
            res = scipy.integrate.trapezoid(res, axis=ax)

        results.append(res.T)

    output = np.concatenate(results, axis=0)

    plt.imshow(output)
    plt.show()

    return output


    # for peak, width in zip(peaks, widths):
    #     gaussian = _gaussian(coords, mu=peak, sigma=width *
    #                          10, ndim=spectrum.ndim - 1)
    #     gaussian = np.expand_dims(gaussian, axis=axis)
    #     filtered = spectrum.array * gaussian
    #
    #     axes = [ax for ax in range(spectrum.ndim) if ax != axis]
    #     axes = sorted(axes, reverse=True)
    #     res = filtered
    #
    #     for ax in axes:
    #         res = np.trapz(res, x=None, axis=ax)
