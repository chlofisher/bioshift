from typing import Self
from numpy.typing import NDArray
from os import PathLike

from bioshift.core.spectrumdatasource import (
    SpectrumDataSource,
    TransformedDataSource,
    SumDataSource,
)
from bioshift.core.spectrumtransform import SpectrumTransform
from bioshift.core.nucleus import NMRNucleus


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
    def data(self) -> NDArray:
        """
        Get an N-dimensional array of data from the data source object.
        SpectrumDataSource implements caching to minimise time spent reading from disk.

        Returns:
            ND array of floating-point spectrum data.
        """
        return self.data_source.get_data()

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

    @classmethod
    def load(cls, path: str | PathLike) -> Self:
        from bioshift.fileio.loadspectrum import load_spectrum
        return load_spectrum(path)

    def add(self, other: Self) -> Self:
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

    def subtract(self, other: Self) -> Self:
        """
        Return a new spectrum equal to the pointwise difference of two spectra.

        Args:
            other: The spectrum to subtract.
        Returns:
            A new Spectrum whose values are the difference of those of the two previous spectra.
        Raises:
            ValueError: If the shapes of the two spectra do not match
        """
        if other.shape != self.shape:
            raise ValueError("Mismatched spectrum dimensions.")

        new_data_source = SumDataSource(
            source1=self.data_source, source2=(-other).data_source
        )

        return Spectrum(
            ndim=self.ndim,
            nuclei=self.nuclei,
            data_source=new_data_source,
            transform=self.transform,
        )

    # def __neg__(self) -> Self:
    #     """
    #     Implements the `-` operator.
    #
    #     Returns:
    #         A new Spectrum with negated values.
    #     """
    #     new_data_source = TransformedDataSource(
    #         parent=self.data_source, func=lambda arr: -arr
    #     )
    #
    #     return Spectrum(
    #         ndim=self.ndim,
    #         nuclei=self.nuclei,
    #         data_source=new_data_source,
    #         transform=self.transform,
    #     )

    def multiply(self, other) -> Self:
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
