from typing import Self
from numpy.typing import NDArray

from bioshift.core.spectrumdatasource import (
    SpectrumDataSource,
    TransformedDataSource,
    SumDataSource,
)
from bioshift.core.spectrumtransform import SpectrumTransform
from bioshift.core.nucleus import NMRNucleus


class Spectrum:
    """NMR spectrum object

    Attributes:
        data_source: Object responsible for lazy-loading and parsing spectrum data.
        nuclei: The type of nucleus (13C, 1H, etc.) associated with each axis.
        transform: Object storing the transformation from array coordinate space to chemical shift space.

    Properties:
        data: N-dimensional numpy array containing the raw spectrum.
        ndim: Number of dimensions of the spectrum.
        shape: Number of data points along each axis of the spectrum.
    """

    name: str
    ndim: int
    nuclei: tuple[NMRNucleus, ...]
    data_source: SpectrumDataSource
    transform: SpectrumTransform

    data: NDArray
    shape: tuple[int, ...]

    def __init__(self, ndim, nuclei, data_source, transform, name=""):
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

    def add(self, other: Self) -> Self:
        """
        Return a new spectrum equal to the pointwise sum of two spectra.

        Args:
            other: The spectrum to add.
        Returns:
            Spectrum: A new spectrum whose values are the sum of those of the two previous spectra.
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
            Spectrum: A new spectrum whose values are the difference of those of the two previous spectra.
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

    def __neg__(self) -> Self:
        """
        Implements the `-` operator.

        Returns: 
            Spectrum: A new spectrum with negated values.
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

    def multiply(self, other) -> Self:
        """
        Return a new spectrum equal to the pointwise product of two spectra.

        Args:
            other: The spectrum to multiply by.
        Returns:
            Spectrum: A new spectrum whose values are the product of those of the two previous spectra.
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

    @property
    def data(self) -> NDArray:
        return self.data_source.get_data()

    @property
    def shape(self) -> NDArray:
        return self.data.shape

    def shift_to_coord(self, shift: NDArray) -> NDArray:
        """
        Convert between chemical shift and grid coordinate systems. Index coordinates are interpolated between integer values.

        Args:
            shift: ND array of chemical shifts.
        Returns:
            NDArray: ND array of grid coordinates.
        """
        return self.transform.inverse.apply(shift)

    def coord_to_shift(self, coord: NDArray) -> NDArray:
        """
        Convert between grid and chemical shift coordinate systems. 

        Args:
            shift: ND array of grid coordinates.
        Returns:
            NDArray: ND array of chemical shifts.
        """
        return self.transform.apply(coord)
