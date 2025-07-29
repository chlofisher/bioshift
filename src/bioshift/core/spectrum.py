from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from enum import Enum
from functools import partial

from bioshift.fileio.spectrumdatasource import (
    SpectrumDataSource, TransformedDataSource, SumDataSource
)
from bioshift.core.spectrumtransform import SpectrumTransform


class NMRNucleus(Enum):
    HYDROGEN = "1H"
    NITROGEN = "15N"
    CARBON = "13C"


class Spectrum:
    """NMR spectrum object.

    Attributes:
        data_source: Object responsible for loading spectrum data from disk.
        params: Object containing spectrum metadata.

    Properties:
        data: N-dimensional numpy array containing the raw spectrum.
        ndim: Number of dimensions of the spectrum.
    """
    ndim: int
    nuclei: tuple[NMRNucleus, ...]
    data_source: SpectrumDataSource
    transform: SpectrumTransform

    data: NDArray

    def __init__(self, ndim, nuclei, data_source, transform):
        self.ndim = ndim
        self.nuclei = nuclei
        self.data_source = data_source
        self.transform = transform

    def __repr__(self):
        return f'Spectrum({self.data.__repr__()})'

    def __add__(self, value: Spectrum) -> Spectrum:
        new_data_source = SumDataSource(
            source1=self.data_source, source2=value.data_source)

        return Spectrum(
            ndim=self.ndim,
            nuclei=self.nuclei,
            data_source=new_data_source,
            transform=self.transform
        )

    def __sub__(self, value: Spectrum) -> Spectrum:
        new_data_source = SumDataSource(
            source1=self.data_source, source2=(-value).data_source)

        return Spectrum(
            ndim=self.ndim,
            nuclei=self.nuclei,
            data_source=new_data_source,
            transform=self.transform
        )

    def __neg__(self) -> Spectrum:
        new_data_source = TransformedDataSource(
            parent=self.data_source,
            func=lambda arr: -arr
        )

        return Spectrum(
            ndim=self.ndim,
            nuclei=self.nuclei,
            data_source=new_data_source,
            transform=self.transform
        )

    @property
    def data(self) -> NDArray:
        return self.data_source.get_data()

    def shift_to_coord(self, shift: NDArray) -> NDArray:
        """Convert chemical shift to array index coordinates.


        Args:
            shift: N-dimensional chemical shift vector.

        Returns:
            N-dimensional spectrum coordinates vector.
        """
        return self.transform.inverse.apply(shift)

    def coord_to_shift(self, coord: NDArray) -> NDArray:
        """Convert array index coordinates to chemical shifts.

        Args:
            shift: N-dimensional spectrum coordinates vector.

        Returns:
            N-dimensional chemical shift vector.
        """
        return self.transform.apply(coord)
