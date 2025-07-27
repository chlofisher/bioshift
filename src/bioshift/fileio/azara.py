from __future__ import annotations
from pathlib import Path
import numpy as np

from bioshift.core.spectrum import NMRNucleus
from bioshift.core.spectrumreference import SpectrumReference
from bioshift.fileio.spectrumreader import SpectrumReader
from bioshift.fileio.blockedspectrum import BlockedSpectrumDataSource


class AzaraSpectrumReader(SpectrumReader):
    """Implementation of SpectrumReader for Azara spectra stored in .par and 
    .spc file pairs.

    Attributes:
        par_path: Path to the .par file.
        spc_path: Path to the .spc file.
    """
    par_path: Path
    spc_path: Path
    params: dict

    def __init__(self, par_path: Path, spc_path: Path):
        self.par_path = par_path
        self.spc_path = spc_path

        self.params = self.get_params()

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

            data_file_param = next(p for p in params if p[0] == 'file')
            spc_file = data_file_param[1].strip()
            potential_spc_paths.append(par_path.parent / Path(spc_file))
        except StopIteration:
            pass

        potential_spc_paths.append(par_path.parent / (par_path.stem + '.spc'))
        potential_spc_paths.append(par_path.parent / par_path.stem)
        potential_spc_paths.append(
            par_path.parent / (Path(par_path.stem).stem + '.spc'))

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

    def get_params(self) -> dict:
        """Read the spectrum metadata from the .par file.

        Returns:
            SpectrumParams object. 
        """
        params_file = AzaraSpectrumReader.parse_par_file(self.par_path)

        params = {}

        # Iterate over all params to get global params
        for param in params_file:
            match param[0]:
                case 'ndim':
                    params['ndim'] = int(param[1])
                case 'file':
                    params['data_file'] = param[1].strip()
                case 'head':
                    params['header_size'] = int(param[1])
                case 'int':
                    params['integer'] = True
                case 'swap':
                    params['swap'] = True
                case 'big_endian':
                    params['endianness'] = False
                case 'little_endian':
                    params['endianness'] = True
                case 'deflate':
                    params['deflate'] = int(params[1])
                case 'reflate':
                    params['reflate'] = int(params[1])
                case 'blocks':
                    raise ValueError("Azara data must be blocked")
                case 'varian':
                    pass  # Ignore this parameter
                case _:
                    pass

        params['header_size'] = 0

        # Record where 'dim' keywords are found, as delimiters of blocks of
        # axis params, including one after the final line.
        dim_positions = [
            i for i, param in enumerate(params_file) if param[0] == 'dim'
        ]
        dim_positions.append(len(params_file))

        ndim = params['ndim']

        shape = [0] * ndim
        block_shape = [0] * ndim
        nuclei = [0] * ndim
        spectral_width = [0] * ndim
        spectrometer_frequency = [0] * ndim
        ref_ppm = [0] * ndim
        ref_coord = [0] * ndim

        # Iterate over each pair of 'dim' positions to get blocks of
        # params associated with each axis.
        for start, stop in zip(dim_positions, dim_positions[1:]):
            for param in params_file[start:stop]:
                match param[0]:
                    case 'dim':
                        dim = int(param[1])
                    case 'npts':
                        shape[dim - 1] = int(param[1])
                    case 'block':
                        block_shape[dim - 1] = int(param[1])
                    case 'sw':
                        spectral_width[dim - 1] = float(param[1])
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
        params['shape'] = tuple(shape[::-1])
        params['block_shape'] = tuple(block_shape[::-1])
        params['nuclei'] = tuple(nuclei[::-1])
        params['spectral_width'] = tuple(spectral_width[::-1])
        params['spectrometer_frequency'] = tuple(spectrometer_frequency[::-1])
        params['ref_ppm'] = tuple(ref_ppm[::-1])
        params['ref_coord'] = tuple(ref_coord[::-1])

        return params

    def get_ndim(self) -> int:
        return self.params['ndim']

    def get_nuclei(self) -> tuple[NMRNucleus, ...]:
        return self.params['nuclei']

    def get_data(self) -> BlockedSpectrumDataSource:
        """Create an AzaraDataSource object for the new spectrum.

        Returns:
            AzaraDataSource object.
        """
        return BlockedSpectrumDataSource(
            path=self.spc_path,
            shape=self.params['shape'],
            block_shape=self.params['block_shape'],
            header_size=self.params['header_size'],
            dtype=np.float32
        )

    def get_reference(self) -> SpectrumReference:
        return SpectrumReference(
            spectrum_shape=self.params['shape'],
            spectral_width=self.params['spectral_width'],
            spectrometer_frequency=self.params['spectrometer_frequency'],
            ref_coord=self.params['ref_coord'],
            ref_ppm=self.params['ref_ppm']
        )

    @classmethod
    def can_read(cls, path: Path) -> bool:
        return path.is_file() and path.suffix in ['.spc', '.par']

    @classmethod
    def parse_par_file(cls, par_path: Path) -> list[tuple[str]]:
        with open(par_path) as file:
            lines = file.readlines()

        # Strip comments and empty lines
        lines = [line for line in lines if line[0] != '!']
        lines = [line for line in lines if line and not line.isspace()]

        return [tuple(line.split(' ')) for line in lines]



