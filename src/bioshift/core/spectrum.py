from __future__ import annotations
from numpy.typing import NDArray
from os import PathLike
from enum import Enum

from bioshift.core.spectrumdatasource import (
    SpectrumDataSource,
    ProjectionDataSource,
    TransformedDataSource,
    SumDataSource,
    SliceDataSource,
)
from bioshift.core.spectrumtransform import SpectrumTransform


class NMRNucleus(Enum):
    H1 = "1H"
    N15 = "15N"
    C13 = "13C"


class Experiment(Enum):
    NHSQC = "NHSQC"
    HNCACB = "HNCACB"


class Spectrum:
    """
    NMR spectrum.

    The recommended way of creating Spectrum instances from spectrum files
    is by using the `Spectrum.load()` function. This automatically determines
    the format of the spectrum and selects the correct SpectrumReader.

    Example usage:
    ```python
    spectrum = Spectrum.load('spectrum_file.ucsf')
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

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == "__call__":
            new_data_source = TransformedDataSource(
                parent=self.data_source, ufunc=ufunc
            )
            return self.__class__(
                ndim=self.ndim,
                nuclei=self.nuclei,
                transform=self.transform,
                data_source=new_data_source,
            )
        else:
            return NotImplemented

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

        from bioshift.fileio.loadspectrum import load_spectrum

        return load_spectrum(path)

    def add(self, other: Spectrum) -> Spectrum:
        """
        Return a new spectrum equal to the pointwise sum of two spectra.

        Args:
            other: The spectrum to add.
        Returns:
            A new Spectrum whose values are the sum of those of the two previous spectra.
        Raises:
            ValueError: If the shapes of the two spectra do not match
        """

        if other.shape != self.shape:
            raise ValueError("Mismatched spectrum dimensions.")

        new_data_source = SumDataSource(
            source1=self.data_source, source2=other.data_source
        )

        return Spectrum(
            ndim=self.ndim,
            nuclei=self.nuclei,
            data_source=new_data_source,
            transform=self.transform,
        )

    def subtract(self, other: Spectrum) -> Spectrum:
        """
        Return a new spectrum equal to the pointwise difference of two spectra.

        Args:
            other: The spectrum to subtract.
        Returns:
            A new Spectrum whose values are the difference of those of the two previous spectra.
        Raises:
            ValueError: If the shapes of the two spectra do not match
        """
        return self.add(-other)

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

    def multiply(self, other) -> Spectrum:
        """
        Return a new spectrum equal to the pointwise product of two spectra.

        Args:
            other: The spectrum to multiply by.
        Returns:
            A new Spectrum whose values are the product of those of the two previous spectra.
        Raises:
            ValueError: If the shapes of the two spectra do not match
        """

        new_data_source = TransformedDataSource(
            parent=self.data_source, func=lambda arr: arr * other
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

        slice_data_source = SliceDataSource(
            parent=self.data_source, axis=axis, level=level
        )

        nuclei = tuple(nuc for i, nuc in enumerate(self.nuclei) if i != axis)

        return Spectrum(
            ndim=self.ndim - 1,
            nuclei=nuclei,
            data_source=slice_data_source,
            transform=self.transform.slice(axis),
        )

    def project(self, axis: int):
        data_source = ProjectionDataSource(
            parent=self.data_source, axis=axis
        )

        nuclei = tuple(nuc for i, nuc in enumerate(self.nuclei) if i != axis)

        return Spectrum(
            ndim=self.ndim - 1,
            nuclei=nuclei,
            data_source=data_source,
            transform=self.transform.slice(axis),
        )





