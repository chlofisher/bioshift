from __future__ import annotations
from pathlib import Path
import struct
import numpy as np
from numpy.typing import NDArray
import math

from protnmr.core.spectrumparams import SpectrumParams
from protnmr.core.spectrumtransform import SpectrumTransform
from protnmr.fileio.spectrumdatasource import SpectrumDataSource
from protnmr.fileio.spectrumreader import SpectrumReader


GLOBAL_HEADER_SIZE = 180
AXIS_HEADER_SIZE = 128


class UCSFSpectrumReader(SpectrumReader):

    def __init__(self, path: Path):
        self.path = path

    @classmethod
    def from_path(cls, path: Path) -> UCSFSpectrumReader:
        """Creates an instance of UCSFSpectrumReader from a given path to a 
        .ucsf file.

        Returns:
            Instance of UCSFSpectrumReader supplied with the path to the 
            .ucsf file.
        """
        return cls(path)

    def get_params(self):
        """Read the spectrum metadata from the .ucsf file header.

        Returns:
            SpectrumParams object.
        """
        with open(self.path, 'rb') as file:
            file.seek(10)
            # Get number of dimensions first to determine header size
            ndim: int = int.from_bytes(file.read(1))

            header_size: int = UCSFSpectrumReader.get_header_size(ndim)

            file.seek(0)
            header_bytes: bytes = file.read(header_size)

        global_header = struct.unpack('>10sBBxB', header_bytes[:14])

        file_type = global_header[0].rstrip(b'\x00').decode('ascii')
        # Ignore global_header[1] as this is the ndim we found earlier
        n_components = int(global_header[2])
        version = int(global_header[3])

        shape = [None] * ndim
        block_shape = [None] * ndim
        n_blocks = [None] * ndim
        nuclei = [None] * ndim
        spectrometer_frequency = [None] * ndim
        spectrum_width = [None] * ndim
        center_shift = [None] * ndim

        for i in range(ndim):
            start = GLOBAL_HEADER_SIZE + i * AXIS_HEADER_SIZE
            stop = GLOBAL_HEADER_SIZE + (i + 1) * AXIS_HEADER_SIZE

            axis_header_bytes = header_bytes[start:stop]
            axis_header = struct.unpack('>6s2xi4xifff', axis_header_bytes[:32])

            nuclei[i] = axis_header[0].rstrip(b'\x00').decode('ascii')
            shape[i] = int(axis_header[1])
            block_shape[i] = int(axis_header[2])
            spectrometer_frequency[i] = float(axis_header[3])
            spectrum_width[i] = float(axis_header[4])
            center_shift[i] = float(axis_header[5])

        shape = np.array(shape)
        block_shape = np.array(block_shape)
        n_blocks = np.array(n_blocks)
        nuclei = np.array(nuclei)

        spectrometer_frequency = np.array(spectrometer_frequency)
        spectrum_width = np.array(spectrum_width)
        center_shift = np.array(center_shift)

        transform = SpectrumTransform.from_reference(
            spectrum_shape=shape,
            spectrum_width=spectrum_width,
            spectrometer_frequency=spectrometer_frequency,
            ref_ppm=center_shift,
            ref_coord=shape/2
        )

        return SpectrumParams(
            ndim=ndim,
            shape=shape,
            block_shape=block_shape,
            n_blocks=n_blocks,
            header_size=header_size,
            nuclei=nuclei,
            transform=transform
        )

    @classmethod
    def get_header_size(cls, ndim: int):
        """Calculates the size of the header in the .ucsf binary file from the 
        number of dimensions in the spectrum.

        Returns:
            Size of the header in bytes.
        """
        return GLOBAL_HEADER_SIZE + AXIS_HEADER_SIZE * ndim

    def get_data(self):
        """Create a UCSFDataSource object for the new spectrum.

        Returns:
            UCSFDataSource object.
        """
        return UCSFDataSource(self.path, self.get_params())

    @classmethod
    def can_read(cls, path: Path) -> bool:
        """Used to determine whether a particular spectrum path can be read 
        using a UCSFSpectrumReader.

        Returns: 
            True if the provided path is a .ucsf file.
        """
        if path.suffix == '.ucsf':
            return True

        return False


class UCSFDataSource(SpectrumDataSource):
    """Concrete implementation of SpectrumDataSource for reading data from 
    .ucsf files.

    Attributes:
        params: Spectrum parameters used to determine how to read the data.
        memmap: Memory map of the file on disk.
    """

    def __init__(self, path: Path, params: SpectrumParams):
        self.params = params
        self.memmap = np.memmap(path,
                                dtype=np.dtype('>f4'),
                                mode='r',
                                offset=params.header_size)
        self.cache = None

    def _load_data(self) -> NDArray:
        """
        Reads the entire spectrum as an ND array of floats.

        Returns:
            NDArray: N-dimensional array of floats containing the spectrum data. 
        """

        blocks = np.empty(self.params.n_blocks, dtype=object)

        for idx in np.ndindex(self.params.n_blocks):
            block_index = self.get_block_index(idx)
            blocks[idx] = self.read_block(block_index)

        data = np.block(blocks.tolist())

        return data

    def read_block(self, index: int) -> NDArray:
        """
        Read a single block from the data file, at the given linear index.

        Args:
            index: Linear index of the block being read in the data file.

        Returns:
            NDArray: N-dimensional array containing the spectrum
                values within the block.
        """
        start: int = index * self.params.block_volume
        end: int = start + self.params.block_volume

        return self.memmap[start:end].reshape(self.params.block_shape)

    def get_block_index(self, idx: tuple[int, ...]) -> int:
        """Takes an ND coordinate index vector and converts it to the linear 
        index of the block within the file.

        Args:
            idx: ND index of the block

        Returns:
            Corresponding linear block index.
        """

        n = self.params.n_blocks

        linear_index = 0
        for axis, index in enumerate(idx):
            linear_index += index * math.prod(n[axis+1:])

        return linear_index
