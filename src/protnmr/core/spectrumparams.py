from dataclasses import dataclass
from enum import Enum
from numpy.typing import NDArray
from math import prod

from protnmr.core.spectrumtransform import SpectrumTransform


class NMRNucleus(Enum):
    HYDROGEN = "1H"
    NITROGEN = "15N"
    CARBON = "13C"


@dataclass
class SpectrumParams:
    """Contains spectrum metadata used to interpret the raw array of data.

    Attributes:
        ndim: Number of dimensions of the spectrum.
        shape: Number of data points along each axis.
        block_shape: Number of data points along each axis per block.
        n_blocks: Number of blocks along each axis.
        transform: Transformation object which maps array coordinates to 
            chemical shifts.
        header_size: Size of the binary data file header in bytes.
    """
    ndim: int
    shape: NDArray[int]
    block_shape: NDArray[int]
    n_blocks: NDArray[int]
    nuclei: tuple[NMRNucleus]
    transform: SpectrumTransform
    header_size: int = 0

    @property
    def block_volume(self) -> int:
        return prod(self.block_shape)
