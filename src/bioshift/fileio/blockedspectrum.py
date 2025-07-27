import numpy as np
import math
from numpy.typing import NDArray
from typing import Optional

from bioshift.fileio.spectrumdatasource import SpectrumDataSource


class BlockedSpectrumDataSource(SpectrumDataSource):
    shape: tuple[int]
    block_shape: tuple[int]
    block_volume: int
    n_blocks: tuple[int]
    header_size: int
    block_volume: int

    memmap: np.memmap
    cache: Optional[NDArray] = None

    def __init__(self, path, shape, block_shape, dtype: np.dtype,
                 header_size):

        self.shape = shape
        self.block_shape = block_shape
        self.block_volume = math.prod(self.block_shape)
        self.n_blocks = tuple(math.ceil(n / k) for n, k in zip(shape, block_shape))

        self.memmap = np.memmap(
            path,
            dtype=dtype,
            mode='r',
            offset=header_size
        )

        self.cache = None

    def _load_data(self) -> NDArray:
        """
        Reads the entire spectrum as an ND array of floats.

        Returns:
            NDArray: N-dimensional array of floats containing the spectrum data.
        """
        blocks = np.empty(self.n_blocks, dtype=object)

        for idx in np.ndindex(tuple(self.n_blocks)):
            block_index = self.get_block_index(idx)
            blocks[idx] = self.read_block(block_index)

        data = np.block(blocks.tolist())

        # Truncate the spectrum to remove padding ispectrum_shapen final block
        slices = tuple(slice(0, n) for n in self.shape)
        data = data[slices]

        print(data.shape)

        return data

    def read_block(self, index: int) -> NDArray:
        """
        Read a single block from the data file, at the given linear index.

        Args:
            index: Linear index of the block being read in the data file.

        Returns:
            NDArray: N-dimensional array containing the spectrum values within 
            the block.
        """
        start: int = index * self.block_volume
        end: int = start + self.block_volume

        return self.memmap[start:end].reshape(self.block_shape)

    def get_block_index(self, idx: tuple[int, ...]) -> int:
        # """Takes an ND coordinate index vector and converts it to the linear 
        # index of the block within the file.
        #
        # Args:
        #     idx: ND index of the block
        #
        # Returns:
        #     Corresponding linear block index.
        # """
        # n = self.n_blocks
        #
        # linear_index = 0
        # for axis, index in enumerate(idx):
        #     linear_index += index * math.prod(n[axis+1:])
        #
        # return linear_index
        return np.ravel_multi_index(idx, self.n_blocks, order='C')
