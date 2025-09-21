import numpy as np
import jax.numpy as jnp
from jax import jit
from numpy.typing import NDArray
from scipy.optimize import curve_fit
from functools import partial

from bioshift.spectra import Spectrum

LN2 = np.log(2.0)


def _gaussian_model(
    x: NDArray,
    *params,
    ndim: int,
    n_peaks: int,
    intensities: NDArray,
) -> NDArray:

    params_array = jnp.array(params)

    k = ndim * n_peaks

    shifts, widths = _unflatten_params(params_array, ndim, n_peaks)
    shifts = params_array[:k].reshape((-1, 1, ndim), order="F")
    widths = params_array[k : 2 * k].reshape((-1, 1, ndim), order="F")

    x_exp = x.transpose()[None, :, :]  # (1, n_datapoints, ndim)
    intensities_exp = jnp.array(intensities[:, None])  # (n_peaks, 1)

    # (n_peaks, n_datapoints)
    # print(f"widths: {np.array(widths)}")
    scaled_magnitude = jnp.sum(((x_exp - shifts) / widths) ** 2, axis=2)
    exponent = LN2 * scaled_magnitude

    mask = exponent <= 5
    masked_exponent = jnp.where(mask, exponent, jnp.inf)

    contributions = intensities_exp * jnp.exp(-masked_exponent)

    output = jnp.sum(contributions, axis=0)  # (n_datapoints,)

    return output


def _jacobian_model(
    x: NDArray, *params, ndim: int, n_peaks: int, intensities: NDArray
) -> NDArray:
    params_array = jnp.array(params)
    x = jnp.array(x).transpose()[None, :, :]

    k = ndim * n_peaks

    peaks = params_array[:k].reshape((-1, 1, ndim), order="F")  # (n_peaks, 1 ndim)
    # (n_peaks, 1, ndim)

    widths = params_array[k : 2 * k].reshape((-1, 1, ndim), order="F")

    intensities_exp = jnp.array(intensities[:, None])  # (n_peaks, 1)

    z = (x - peaks) / widths

    # (n_peaks, n_datapoints)
    exponent = LN2 * jnp.sum(jnp.square(z), axis=2)

    mask = exponent <= 5

    contributions = intensities_exp * jnp.where(mask, jnp.exp(-exponent), 0)

    df_dexp = contributions[:, :, None]

    dexp_dmu = 2 * LN2 * z / widths
    dexp_dgamma = 2 * LN2 * jnp.square(z) / widths

    df_dmu = df_dexp * dexp_dmu
    df_dgamma = df_dexp * dexp_dgamma

    df_dmu_flat = df_dmu.transpose(2, 0, 1).reshape(ndim * n_peaks, -1)
    df_dgamma_flat = df_dgamma.transpose(2, 0, 1).reshape(ndim * n_peaks, -1)

    jacobian = jnp.concatenate((df_dmu_flat, df_dgamma_flat), axis=0).T

    return jacobian


def _flatten_params(shifts: NDArray, widths: NDArray) -> NDArray:
    return np.concatenate(shifts.ravel(), widths.ravel())


def _unflatten_params(
    params: NDArray, ndim: int, n_peaks: int
) -> tuple[NDArray, NDArray]:

    k = ndim * n_peaks
    shifts = params[:k].reshape((-1, ndim), order="F")
    widths = params[k : 2 * k].reshape((-1, ndim), order="F")

    return shifts, widths


def fit(
    spectrum: Spectrum,
    initial_peaks: NDArray,
    max_shift_change: tuple[int],
) -> NDArray:
    n_peaks = len(initial_peaks)

    transform = spectrum.transform
    ndim = transform.ndim

    axes = [
        np.linspace(min, max, s)
        for s, min, max in zip(
            transform.shape, transform.bounds[0], transform.bounds[1]
        )
    ]
    mesh = np.meshgrid(*axes, indexing="ij")

    xdata = np.vstack([m.ravel() for m in mesh])
    ydata = spectrum.array.ravel()

    xdata_decimated = xdata[:, ::10]
    ydata_decimated = ydata[::10]

    initial_shifts = initial_peaks[:, 0].T.ravel()
    initial_widths = initial_peaks[:, 1].T.ravel()

    p0 = np.concatenate((initial_shifts, initial_widths))

    pos_min = initial_shifts - np.concatenate(
        [np.ones(n_peaks) * delta for delta in max_shift_change]
    )
    pos_max = initial_shifts + np.concatenate(
        [np.ones(n_peaks) * delta for delta in max_shift_change]
    )

    width_min = np.zeros(ndim * n_peaks)
    width_max = pos_max - pos_min

    min = np.concatenate((pos_min, width_min))
    max = np.concatenate((pos_max, width_max))

    bounds = (min, max)

    intensities = np.ones(n_peaks)

    model = jit(
        partial(
            _gaussian_model,
            ndim=ndim,
            n_peaks=n_peaks,
            intensities=intensities,
        )
    )

    jacobian = jit(
        partial(
            _jacobian_model,
            ndim=ndim,
            n_peaks=n_peaks,
            intensities=intensities,
        )
    )

    popt, pcov = curve_fit(
        model, xdata_decimated, ydata_decimated, p0=p0, bounds=bounds, jac=jacobian
    )

    # fitted_spectrum = model(xdata, *popt).reshape(512, 1024)
    # residual = (spectrum.data - fitted_spectrum) ** 2

    k = ndim * n_peaks
    shifts = popt[:k].reshape((-1, ndim), order="F")
    widths = popt[k : 2 * k].reshape((-1, ndim), order="F")

    return shifts, widths
