from numpy.typing import NDArray

from bioshift.core.spectrumparams import SpectrumParams
from bioshift.fileio.spectrumdatasource import SpectrumDataSource


class Spectrum:
    """NMR spectrum object.

    Attributes:
        data_source: Object responsible for loading spectrum data from disk.
        params: Object containing spectrum metadata.

    Properties:
        data: N-dimensional numpy array containing the raw spectrum.
        ndim: Number of dimensions of the spectrum.
    """
    data_source: SpectrumDataSource
    params: SpectrumParams

    data: NDArray
    ndim: int

    def __init__(self, params: SpectrumParams, data_source: SpectrumDataSource):
        self.params = params
        self.data_source = data_source

    def __repr__(self):
        return f'Spectrum({self.data.__repr__()})'

    @property
    def ndim(self) -> int:
        return self.params.ndim

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
        return self.params.transform.inverse.apply(shift)

    def coord_to_shift(self, coord: NDArray) -> NDArray:
        """Convert array index coordinates to chemical shifts.

        Args:
            shift: N-dimensional spectrum coordinates vector.

        Returns:
            N-dimensional chemical shift vector.
        """
        return self.params.transform.apply(coord)
