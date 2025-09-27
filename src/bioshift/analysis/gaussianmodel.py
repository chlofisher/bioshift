from typing import Optional
from functools import partial

import numpy as np
from numpy.typing import NDArray

from scipy.optimize import curve_fit

import jax
from jax import numpy as jnp

from bioshift.spectra import Spectrum


LN2 = np.log(2.0)


def _gaussian_model(x: jax.Array, *params, ndim: int, n_peaks: int) -> jax.Array:
    params_array = jnp.array(params)

    k = ndim * n_peaks

    # (n_peaks, 1, ndim)
    shifts = params_array[:k].reshape((-1, 1, ndim), order="F") 
    widths = params_array[k : 2 * k].reshape((-1, 1, ndim), order="F")

    # (n_peaks, 1)
    intensities = params_array[2 * k :, None]

    # (1, n_datapoints, ndim)
    x = x.transpose()[None, :, :]  
    z = (x - shifts) / widths

    # (n_peaks, n_datapoints)
    exponent = LN2 * jnp.sum(jnp.square(z), axis=2)

    mask = exponent <= 5
    masked_exponent = jnp.where(mask, exponent, jnp.inf)

    contributions = intensities * jnp.exp(-masked_exponent)
    return jnp.sum(contributions, axis=0)  # (n_datapoints,)


def _jacobian(x: jax.Array, *params, ndim: int, n_peaks: int) -> jax.Array:
    params_array = jnp.array(params)

    k = ndim * n_peaks

    # (n_peaks, 1, ndim)
    shifts = params_array[:k].reshape((-1, 1, ndim), order="F")
    widths = params_array[k : 2 * k].reshape((-1, 1, ndim), order="F")
    intensities = params_array[2 * k :, None]

    # (n_peaks, 1)
    x = x.transpose()[None, :, :]
    z = (x - shifts) / widths

    # (n_peaks, n_datapoints)
    exponent = LN2 * jnp.sum(jnp.square(z), axis=2)

    mask = exponent <= 5
    masked_exponent = jnp.where(mask, exponent, jnp.inf)

    contributions = intensities * jnp.exp(-masked_exponent)

    df_dexp = contributions[:, :, None]

    dexp_dmu = 2 * LN2 * z / widths
    dexp_dgamma = 2 * LN2 * jnp.square(z) / widths

    df_dmu = df_dexp * dexp_dmu
    df_dgamma = df_dexp * dexp_dgamma
    df_dintensity = contributions

    df_dmu_flat = df_dmu.transpose(2, 0, 1).reshape(ndim * n_peaks, -1)
    df_dgamma_flat = df_dgamma.transpose(2, 0, 1).reshape(ndim * n_peaks, -1)

    return jnp.concatenate((df_dmu_flat, df_dgamma_flat, df_dintensity), axis=0).T


def _flatten_params(
    shifts: NDArray,
    widths: NDArray,
    intensities: NDArray
) -> NDArray:
    return jnp.concatenate([shifts.ravel(), widths.ravel(), intensities])


def _unflatten_params(
    params: NDArray, ndim: int, n_peaks: int
) -> tuple[NDArray, NDArray]:

    k = ndim * n_peaks
    shifts = params[:k].reshape((-1, ndim), order="F")
    widths = params[k : 2 * k].reshape((-1, ndim), order="F")
    intensities = params[2 * k :]

    return shifts, widths, intensities


def _axes(samples: tuple[int], bounds: NDArray) -> tuple[NDArray, ...]:
    axes = tuple(
        np.linspace(min, max, n) for n, min, max in zip(samples, bounds[0], bounds[1])
    )
    return axes


def _grid(samples: tuple[int], bounds: NDArray) -> tuple[NDArray, ...]:
    axes = _axes(samples, bounds)
    return np.meshgrid(*axes, indexing="ij")


class GaussianPeakModel:
    def __init__(
        self,
        ndim: int,
        n_peaks_init: int,
        shifts_init: Optional[NDArray] = None,
        widths_init: Optional[NDArray] = None,
        intensities_init: Optional[NDArray] = None,
    ):
        n_peaks_match = True
        for arr in (shifts_init, widths_init, intensities_init):
            if arr is not None:
                n_peaks_match = n_peaks_match and arr.shape[0] == n_peaks_init

        if not n_peaks_match:
            raise ValueError("Mismatched lengths of initial parameter arrays.")

        self.n_peaks_init = n_peaks_init
        self.shifts_init = shifts_init
        self.widths_init = widths_init
        self.intensities_init = intensities_init

    def _xgrid(self, spectrum, region=None, samples=None) -> NDArray:
        if region is not None and samples is None:
            raise ValueError(
                (
                    "Must provide number of samples along each axis "
                    "when specifying a bounding region."
                )
            )

        if region is not None and samples is not None:
            grid = _grid(samples, region)

        if region is None and samples is None:
            grid = spectrum.transform.grid

        if region is None and samples is not None:
            grid = _grid(samples, spectrum.transform.bounds)

        return grid

    def _ygrid(self, spectrum, x_grid) -> NDArray:
        return spectrum.intensity(x_grid)

    def fit(
        self,
        spectrum: Spectrum,
        region: Optional[NDArray] = None,
        samples: Optional[NDArray] = None,
    ):
        spectrum = spectrum.normalize()

        ndim = spectrum.ndim
        n_peaks = self.n_peaks_init

        x_grid = self._xgrid(spectrum, region, samples)
        y_grid = self._ygrid(spectrum, x_grid)

        xdata = np.vstack([m.ravel() for m in x_grid])
        ydata = y_grid.ravel()

        if region is None:
            region = spectrum.transform.bounds

        if self.intensities_init is None:
            if self.shifts_init is None:
                self.intensities_init = np.ones(n_peaks) * 0.5
            else:
                self.intensities_init = spectrum.intensity(self.shifts_init)

        intensity_min = np.zeros(n_peaks)
        intensity_max = np.ones(n_peaks)

        if self.shifts_init is None:
            random = np.random.rand(n_peaks, ndim)
            self.shifts_init = region[1] + (region[0] - region[1]) * random

        shift_min = np.repeat(region[1], n_peaks)
        shift_max = np.repeat(region[0], n_peaks)

        if self.widths_init is None:
            random = np.random.rand(n_peaks, ndim)
            self.widths_init = np.array([0.1, 0.1, 0.02]) * (1 + random)

        width_min = np.zeros(n_peaks * ndim)
        width_max = np.concat([np.ones(n_peaks), np.ones(n_peaks), np.ones(n_peaks) * 0.1])

        p0 = np.concatenate(
            [
                self.shifts_init.T.ravel(),
                self.widths_init.T.ravel(),
                self.intensities_init.T.ravel(),
            ]
        )

        min = np.concatenate((shift_min, width_min, intensity_min))
        max = np.concatenate((shift_max, width_max, intensity_max))

        param_bounds = (min, max)

        model = partial(
            _gaussian_model,
            ndim=ndim,
            n_peaks=n_peaks,
        )

        jacobian = partial(
            _jacobian,
            ndim=ndim,
            n_peaks=n_peaks,
        )

        model = jax.jit(model)
        jacobian = jax.jit(jacobian)

        popt, pcov = curve_fit(
            model, xdata, ydata, p0=p0, bounds=param_bounds, jac=jacobian, method="trf"
        )

        shifts, widths, intensities = _unflatten_params(popt, ndim, n_peaks)

        self.shifts = shifts
        self.widths = widths
        self.intensities = intensities

    def residual(self, spectrum):
        params = _flatten_params(self.shifts, self.widths, self.intensities)
        return _gaussian_model(spectrum.transform.grid, *params, spectrum.ndim, n_peaks=len(self.intensities))
