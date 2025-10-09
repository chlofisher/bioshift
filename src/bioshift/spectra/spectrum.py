from __future__ import annotations

import math
from os import PathLike
from enum import Enum
from functools import partial

import numpy as np
from numpy.typing import NDArray
import skimage
from scipy.interpolate import RegularGridInterpolator

from bioshift.spectra.spectrumdatasource import (
    SpectrumDataSource,
    TransformedDataSource,
)
from bioshift.spectra.spectrumtransform import SpectrumTransform


# class NMRNucleus(Enum):
#     H1 = "H"
#     N15 = "N"
#     C13 = "C"


class NMRExperiment(Enum):
    NHSQC = "NHSQC"
    HNCO = "HNCO"
    HNCACB = "HNCACB"
    HNCA = "HNCA"
    HNCOCACB = "HN(CO)CACB"


class Spectrum:
    """
    NMR spectrum.

    The recommended way of creating Spectrum instances from spectrum files
    is by using the `Spectrum.load()` function. This automatically determines
    the format of the spectrum and selects the correct SpectrumReader.

    Example:
    ```python
    spectrum = Spectrum.load('./spectrum_file.ucsf')
    ```
    """

    ndim: int
    """Number of dimensions of the spectrum."""

    nuclei: tuple[str, ...]
    """The type of nucleus (13C, 1H, etc.) associated with each axis."""

    data_source: SpectrumDataSource
    """Object responsible for lazy-loading and parsing spectrum data."""

    transform: SpectrumTransform
    """Object storing the transformation from array coordinate space to chemical shift space."""

    experiment: NMRExperiment
    """Enum value for the type of NMR experiment the spectrum is from (e.g., HSQC, HNCACB)"""

    @property
    def shape(self) -> NDArray:
        """
        Returns:
            Shape of the underlying data array.
        """
        return self.transform.shape

    def __init__(
        self,
        ndim: int,
        nuclei: tuple[NMRNucleus, ...],
        data_source: SpectrumDataSource,
        transform: SpectrumTransform,
    ):
        self.ndim = ndim
        self.nuclei = nuclei
        self.data_source = data_source
        self.transform = transform

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  axes={str(self.nuclei)},\n"
            f"  source={self.data_source.__repr__()},\n"
            f")"
        )

    @property
    def array(self) -> NDArray:
        """
        Get the raw data of the spectrum as a numpy array.
        """
        return self.data_source.get_data()

    @classmethod
    def load(cls, path: str | PathLike) -> Spectrum:
        """
        Create a spectrum from a path to a spectrum file.
        Automatically determines the file format and dispatches the correct spectrum reader.
        Currently spectra stored in .ucsf and .spc(.par) files are supported.

        Args:
            path: Path to the spectrum file.
        Returns:
            Spectrum object
        """
        from bioshift.io import load_spectrum

        return load_spectrum(path)

    def __neg__(self) -> Spectrum:
        """
        Implements the `-` operator.

        Returns:
            A new Spectrum with negated values.
        """
        new_data_source = TransformedDataSource(
            parent=self.data_source, func=lambda arr: -arr
        )

        return Spectrum(
            ndim=self.ndim,
            nuclei=self.nuclei,
            data_source=new_data_source,
            transform=self.transform,
        )

    def intensity(self, shift: NDArray | tuple[NDArray, ...]) -> NDArray:
        """
        Evaluate the interpolated intensity at given chemical shift coords.

        This method wraps `scipy.interpolate.RegularGridInterpolator` to interpolate
        values from a regularly spaced ND grid (`self.array`) defined over axes
        stored in `self.transform.axes`.
        Args:
            shift: Coordinates at which to evaluate the intensity. 
            Accepts:
                - A NumPy array of shape (n_points, ndim), where each row is a point.
                - A tuple of arrays (e.g., from `np.meshgrid`) of equal shape, which
                  will be stacked into coordinate points.
        Returns:
            Interpolated intensity values at the specified coordinates.
            If `shift` is a tuple of meshgrid arrays, the output shape matches the grid shape.
        """

        interp = RegularGridInterpolator(
            points=self.transform.axes,
            values=self.array,
            method="linear",
            bounds_error=False,
            fill_value=0,
        )
        return interp(shift)

    def slice(self, axis: int, z: float) -> Spectrum:
        """
        Take a slice from the spectrum along the specified axis.

        Args:
            axis: The index of the axis perpendicular to the slice plane.
            z: The chemical shift of the plane along the specified axis.
        Returns:
            A new spectrum, with one fewer dimension, corresponding to the slice plane.
        """
        level = (
            z * self.transform.inverse_scaling[axis]
            + self.transform.inverse_offset[axis]
        )

        def slice_func(arr: NDArray):
            floor = (math.floor(level),)
            ceil = (math.ceil(level),)
            frac = level - floor

            below = arr.take(floor, axis=axis).squeeze(axis)
            above = arr.take(ceil, axis=axis).squeeze(axis)

            return below * (1 - frac) + above * frac

        slice_data_source = TransformedDataSource(
            parent=self.data_source, func=slice_func
        )

        nuclei = tuple(nuc for i, nuc in enumerate(self.nuclei) if i != axis)

        return Spectrum(
            ndim=self.ndim - 1,
            nuclei=nuclei,
            data_source=slice_data_source,
            transform=self.transform.slice(axis),
        )

    def project(self, axis: int) -> Spectrum:
        """
        Project the spectrum along one of its coordinate axes by taking the integral.

        Args:
            axis: The axis along which to project the spectrum.
        Returns:
            A new spectrum with one fewer dimension.
        """
        data_source = TransformedDataSource(
            parent=self.data_source, func=partial(np.trapz, axis=axis)
        )

        nuclei = tuple(nuc for i, nuc in enumerate(self.nuclei) if i != axis)

        return Spectrum(
            ndim=self.ndim - 1,
            nuclei=nuclei,
            data_source=data_source,
            transform=self.transform.slice(axis),
        )

    def blur(self, sigma: tuple[float]) -> Spectrum:
        """
        Apply a gaussian blur to the spectrum.
        
        Args:
            sigma: The standard deviation of the gaussian kernel used for the blur.
        Returns:
            A new spectrum which has been blurred.
        """
        if len(sigma) != self.ndim:
            raise ValueError(
                f"""Mismatched dimensions. Spectrum is {self.ndim}D,
                but a {len(sigma)}D vector of sigmas was provided."""
            )

        sigma_scaled = np.abs(np.array(sigma) / self.transform.scaling)
        gaussian_func = partial(skimage.filters.gaussian, sigma=sigma_scaled)

        data_source = TransformedDataSource(parent=self.data_source, func=gaussian_func)

        return Spectrum(
            ndim=self.ndim,
            nuclei=self.nuclei,
            data_source=data_source,
            transform=self.transform,
        )

    def threshold(self, level: float, soft: bool=True) -> Spectrum:
        """
        Apply a threshold to the spectrum.

        Args:
            level: The threshold level. All points with intensities below this value are set to zero.
            soft: If set to True, values above the threshold are shifted down by the threshold level,
            in order to keep the spectrum continuous.
        Returns:
            A new spectrum with all points below the threshold value set to zero.
        """
        if soft:
            def threshold_func(arr: NDArray) -> NDArray:
                return np.where(np.abs(arr) < level, 0, arr)
        else:
            def threshold_func(arr: NDArray) -> NDArray:
                return np.where(np.abs(arr) < level, 0, arr - level * np.sign(arr))

        data_source = TransformedDataSource(
            parent=self.data_source, func=threshold_func
        )

        return Spectrum(
            ndim=self.ndim,
            nuclei=self.nuclei,
            data_source=data_source,
            transform=self.transform,
        )

    def normalize(self) -> Spectrum:
        """
        Normalize the spectrum to have maximum 1.0.
        Returns:
            A new spectrum with all values divided by the maximum of the original.
        """
        def normalize_func(arr: NDArray) -> NDArray:
            max = np.abs(arr).max()
            return arr / max

        data_source = TransformedDataSource(
            parent=self.data_source, func=normalize_func
        )

        return Spectrum(
            ndim=self.ndim,
            nuclei=self.nuclei,
            data_source=data_source,
            transform=self.transform,
        )

    def transpose(self, axes: tuple[int, ...] | None=None) -> Spectrum:
        """
        Transpose the spectrum, swapping the ordering of the axes.
        Args:
            Axes: Tuple of integers defining the new ordering of the axes. 
            If None are provided, then the default behaviour is to invert 
            the ordering of the axes (the same as np.transpose)
        Returns:
            A new spectrum with axes reordered.
        """
        if axes is None:
            axes = range(self.ndim)[::-1]

        transpose_func = partial(np.transpose, axes=axes)
        data_source = TransformedDataSource(
            parent=self.data_source, func=transpose_func
        )

        new_nuclei = tuple(self.nuclei[ax] for ax in axes)
        new_transform = self.transform.transpose(axes)

        return Spectrum(
            ndim=self.ndim,
            nuclei=new_nuclei,
            data_source=data_source,
            transform=new_transform,
        )
