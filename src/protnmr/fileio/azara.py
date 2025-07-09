from __future__ import annotations
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
import math

from protnmr.fileio.spectrumreader import SpectrumReader
from protnmr.fileio.spectrumdatasource import SpectrumDataSource
from protnmr.core.spectrumparams import SpectrumParams
from protnmr.core.spectrumtransform import SpectrumTransform


class AzaraSpectrumReader(SpectrumReader):
    """Implementation of SpectrumReader for Azara spectra stored in .par and 
    .spc file pairs.

    Attributes:
        par_path: Path to the .par file.
        spc_path: Path to the .spc file.
    """
    par_path: Path
    spc_path: Path

    def __init__(self, par_path: Path, spc_path: Path):
        self.par_path = par_path
        self.spc_path = spc_path

    @classmethod
    def from_path(cls, path: Path) -> AzaraSpectrumReader:
        """Creates an instance of AzaraSpectrumReader from a given .spc or 
        .par path. Given a .par file it will find the corresponding .spc and 
        vice-versa. 

        Returns:
            Instance of AzaraSpectrumReader supplied with paths for both the 
            .spc and .par files.

        """
        if path.suffix == '.par':
            par_path = path
            spc_path = cls.spc_from_par(par_path)
        elif path.suffix == '.spc':
            spc_path = path
            par_path = cls.par_from_spc(spc_path)

        return cls(par_path, spc_path)

    @classmethod
    def spc_from_par(cls, par_path: Path) -> Path:
        """Finds a corresponding .spc file from a .par file. First checks for 
        a .spc file specified in the .par file, failing that checks for a .spc 
        with a matching name.

        Args:
            par_path: Path to the .par file.

        Returns:
            Path to a .spc file.

        Raises:
            FileNotFoundError: if a .spc file can not be found
        """
        potential_spc_paths = []

        try:
            # Need to peek ahead at the .par file before fully loading the
            # params to check for a .spc file
            params = AzaraSpectrumReader.parse_par_file(par_path)

            data_file_param = next(p for p in params if p[0] == 'data_file')
            spc_file = data_file_param[1].strip()
            potential_spc_paths.append(par_path.parent / Path(spc_file))
        except StopIteration:
            pass

        potential_spc_paths.append(par_path.parent / par_path.stem + '.spc')

        for spc_path in potential_spc_paths:
            if spc_path.is_file():
                return spc_path

        raise FileNotFoundError(
            f'Could not find .spc file corresponding to {par_path}')

    @classmethod
    def par_from_spc(cls, spc_path: Path) -> Path:
        """Finds a corresponding .par file from a .spc file. Given 
        spectrum.spc, checks spectrum.spc.par and spectrum.par.

        Args:
            spc_path: Path to the .spc file.

        Returns:
            Path to a .par file.

        Raises:
            FileNotFoundError: if a .par file can not be found
        """
        potential_par_paths = []

        # e.g. spectrum.spc.par
        potential_par_paths.append(spc_path.parent / (spc_path.name + '.par'))

        # e.g. spectrum.par
        potential_par_paths.append(spc_path.parent / (spc_path.stem + '.par'))

        for par_path in potential_par_paths:
            if par_path.is_file():
                return par_path

        raise FileNotFoundError(
            f'Could not find valid .par file corresponding to {spc_path}')

    def get_params(self) -> SpectrumParams:
        """Read the spectrum metadata from the .par file.

        Returns:
            SpectrumParams object. 
        """
        params = AzaraSpectrumReader.parse_par_file(self.par_path)

        # Iterate over all params to get global params
        for param in params:
            match param[0]:
                case 'ndim':
                    ndim = int(param[1])
                # case 'file':
                #     data_file = param[1].strip()
                case 'head':
                    header_size = int(param[1])
                # case 'int':
                #     integer = True
                # case 'swap':
                #     swap = True
                # case 'big_endian':
                #     endianness = False
                # case 'little_endian':
                #     endianness = True
                # case 'deflate':
                #     deflate = int(params[1])
                # case 'reflate':
                #     reflate = int(params[1])
                case 'blocks':
                    raise ValueError("Azara data must be blocked")
                case 'varian':
                    pass  # Ignore this parameter
                case _:
                    pass

        # Record where 'dim' keywords are found, as delimiters of blocks of
        # axis params, including one after the final line.
        dim_positions = [
            i for i, param in enumerate(params) if param[0] == 'dim'
        ]
        dim_positions.append(len(params))

        shape = [0] * ndim
        block_shape = [0] * ndim
        nuclei = [0] * ndim
        spectrum_width = [0] * ndim
        spectrometer_frequency = [0] * ndim
        ref_ppm = [0] * ndim
        ref_coord = [0] * ndim

        # Iterate over each pair of 'dim' positions to get blocks of
        # params associated with each axis.
        for start, stop in zip(dim_positions, dim_positions[1:]):
            for param in params[start:stop]:
                match param[0]:
                    case 'dim':
                        dim = int(param[1])
                    case 'npts':
                        shape[dim - 1] = int(param[1])
                    case 'block':
                        block_shape[dim - 1] = int(param[1])
                    case 'sw':
                        spectrum_width[dim - 1] = float(param[1])
                    case 'sf':
                        spectrometer_frequency[dim - 1] = float(param[1])
                    case 'refppm':
                        ref_ppm[dim - 1] = float(param[1])
                    case 'refpt':
                        ref_coord[dim - 1] = float(param[1])
                    case 'nuc':
                        nuclei[dim - 1] = param[1].strip()
                    case 'params':
                        pass
                    case 'sigmas':
                        pass
                    case _:
                        pass

        # Invert to row-major order
        shape = np.array(shape[::-1])
        block_shape = np.array(block_shape[::-1])
        n_blocks = np.floor_divide(shape, block_shape)
        nuclei = np.array(nuclei[::-1])
        spectrum_width = np.array(spectrum_width[::-1])
        spectrometer_frequency = np.array(spectrometer_frequency[::-1])
        ref_ppm = np.array(ref_ppm[::-1])
        ref_coord = np.array(ref_coord[::-1])

        # Create a SpectrumTransform object from the given reference
        transform = SpectrumTransform.from_reference(
            spectrum_shape=shape,
            spectrum_width=spectrum_width,
            spectrometer_frequency=spectrometer_frequency,
            ref_ppm=ref_ppm,
            ref_coord=ref_coord
        )

        return SpectrumParams(
            ndim=ndim,
            shape=shape,
            block_shape=block_shape,
            n_blocks=n_blocks,
            header_size=0,
            nuclei=nuclei,
            transform=transform
        )

    def get_data(self) -> AzaraDataSource:
        """Create an AzaraDataSource object for the new spectrum.

        Returns:
            AzaraDataSource object.
        """
        return AzaraDataSource(self.spc_path, self.get_params())

    @classmethod
    def can_read(cls, path: Path) -> bool:
        """Used to determine whether a particular spectrum path should be read 
        using an AzaraSpectrumReader.

        Returns: 
            True if the provided path is a .spc or .par file.
        """
        return path.is_file() and path.suffix in ['.spc', '.par']

    @classmethod
    def parse_par_file(cls, par_path: Path) -> list[tuple[str]]:
        with open(par_path) as file:
            lines = file.readlines()

        # Strip comments and empty lines
        lines = [line for line in lines if line[0] != '!']
        lines = [line for line in lines if line and not line.isspace()]

        return [tuple(line.split(' ')) for line in lines]


class AzaraDataSource(SpectrumDataSource):
    """Concrete implementation of SpectrumDataSource for reading data from 
    Azara .spc files.

    Attributes:
        params: Spectrum parameters used to determine how to read the data.
        memmap: Memory map of the file on disk.
    """
    params: SpectrumParams
    memmap: np.memmap

    def __init__(self, path: Path, params: SpectrumParams):
        self.params = params
        self.memmap = np.memmap(path,
                                dtype=np.float32,
                                mode='r',
                                offset=params.header_size)
        self.cache: NDArray = None

    def _load_data(self) -> NDArray:
        """
        Reads the entire spectrum as an ND array of floats.

        Returns:
            NDArray: N-dimensional array of floats containing the spectrum data.
        """
        blocks = np.empty(self.params.n_blocks, dtype=object)

        for idx in np.ndindex(tuple(self.params.n_blocks)):
            block_index = self.get_block_index(idx)
            blocks[idx] = self.read_block(block_index)

        data = np.block(blocks.tolist())

        # Truncate the spectrum to remove padding in final block
        slices = tuple(slice(0, n) for n in self.params.shape)
        data = data[slices]

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

        # match self.params.ndim:
        #     case 2:
        #         return idx[1] + n[1] * idx[0]
        #     case 3:
        #         return idx[2] + n[2] * idx[1] + n[2] * n[1] * idx[0]
        #     case 4:
        #         return (idx[3] + n[3] * idx[2] + n[3] * n[2] * idx[1]
        #                 + n[3] * n[2] * n[1] * idx[0])
        #
        # raise NotImplementedError(
        #     'Currently only 2D and 3D spectra are supported.')
