from __future__ import annotations
from numpy.typing import NDArray
from os import PathLike
from enum import Enum
import math
import numpy as np
from functools import partial
import skimage

from bioshift.spectra.spectrumdatasource import (
    SpectrumDataSource,
    TransformedDataSource,
)
from bioshift.spectra.spectrumtransform import SpectrumTransform


class NMRNucleus(Enum):
    H1 = "1H"
    N15 = "15N"
    C13 = "13C"


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

    Example usage:
    ```python
    spectrum = Spectrum.load('./spectrum_file.ucsf')
    ```
    """

    ndim: int
    """Number of dimensions of the spectrum."""

    nuclei: tuple[NMRNucleus, ...]
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

    def __init__(self, ndim, nuclei, data_source, transform):
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

    def __array__(self, dtype=None, copy=None):
        return self.data_source.get_data()

    @property
    def array(self) -> NDArray:
        return self.__array__()

    @classmethod
    def load(cls, path: str | PathLike) -> Spectrum:
        """
        Create a spectrum from a path to a spectrum file.
        Automatically determines the file format and dispatches the correct spectrum reader.

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

    def slice(self, axis: int, z: float):
        """
        Take a slice from the spectrum along the specified axis.

        Args:
            axis: The index of the axis perpendicular to the slice plane.
            z: The chemical shift of the plane along the specified axis.
        Returns:
            A new spectrum, with one fewer dimension.
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

    def project(self, axis: int):
        project_func = partial(np.trapz, axis=axis)
        data_source = TransformedDataSource(parent=self.data_source, func=project_func)

        nuclei = tuple(nuc for i, nuc in enumerate(self.nuclei) if i != axis)

        return Spectrum(
            ndim=self.ndim - 1,
            nuclei=nuclei,
            data_source=data_source,
            transform=self.transform.slice(axis),
        )

    def blur(self, sigma: tuple[float]):
        if len(sigma) != self.ndim:
            raise ValueError(
                f"""Mismatched dimensions. Spectrum is {input.ndim}D,
                but a {len(sigma)}D vector of sigmas was provided."""
            )

        sigma_scaled = np.array(sigma) / input.transform.scaling
        gaussian_func = partial(skimage.filters.gaussian, sigma=sigma_scaled)

        data_source = TransformedDataSource(parent=self.data_source, func=gaussian_func)

        return Spectrum(
            ndim=self.ndim,
            nuclei=self.nuclei,
            data_source=data_source,
            transform=self.transform,
        )

    def threshold(self, level: float):
        def threshold_func(arr: NDArray):
            return np.where(np.abs(arr) < level, 0, arr)

        data_source = TransformedDataSource(
            parent=self.data_source, func=threshold_func
        )

        return Spectrum(
            ndim=self.ndim,
            nuclei=self.nuclei,
            data_source=data_source,
            transform=self.transform,
        )

    def normalize(self):
        def normalize_func(arr: NDArray):
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

    def transpose(self, axes=None):
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
            transform=new_transform
        )

        
