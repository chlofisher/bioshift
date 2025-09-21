from __future__ import annotations
from warnings import warn
from typing import Any
from pathlib import Path
import numpy as np

from bioshift.spectra import SpectrumTransform, NMRNucleus
from bioshift.io.spectrumreader import SpectrumReader 
from bioshift.io.blockedspectrum import BlockedSpectrumDataSource


def _spc_from_par(par_path: Path) -> Path:
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
        params = _parse_par_file(par_path)

        data_file_param = next(p for p in params if p[0] == "file")
        spc_file = data_file_param[1].strip()
        potential_spc_paths.append(par_path.parent / Path(spc_file))
    except StopIteration:
        pass

    potential_spc_paths.append(par_path.parent / (par_path.stem + ".spc"))
    potential_spc_paths.append(par_path.parent / par_path.stem)
    potential_spc_paths.append(par_path.parent / (Path(par_path.stem).stem + ".spc"))

    for spc_path in potential_spc_paths:
        if spc_path.is_file():
            return spc_path

    raise FileNotFoundError(f"Could not find .spc file corresponding to {par_path}")


def _par_from_spc(spc_path: Path) -> Path:
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
    potential_par_paths.append(spc_path.parent / (spc_path.name + ".par"))

    # e.g. spectrum.par
    potential_par_paths.append(spc_path.parent / (spc_path.stem + ".par"))

    for par_path in potential_par_paths:
        if par_path.is_file():
            return par_path

    raise FileNotFoundError(
        f"Could not find valid .par file corresponding to {spc_path}"
    )


def _parse_par_file(par_path: Path) -> list[tuple[str]]:
    with open(par_path) as file:
        lines = file.readlines()

    # Strip comments and empty lines
    lines = [line for line in lines if line[0] != "!"]
    lines = [line for line in lines if line and not line.isspace()]

    return [tuple(line.split(" ", 1)) for line in lines]


class AzaraSpectrumReader(SpectrumReader):
    par_path: Path
    spc_path: Path
    _params: dict

    def __init__(self, par_path: Path, spc_path: Path):
        self.par_path = par_path
        self.spc_path = spc_path

        self._params = self._get_params()

    @classmethod
    def from_path(cls, path: Path) -> AzaraSpectrumReader:
        if path.suffix == ".par":
            par_path = path
            spc_path = _spc_from_par(par_path)
        elif path.suffix == ".spc":
            spc_path = path
            par_path = _par_from_spc(spc_path)

        return cls(par_path, spc_path)

    # TODO Break up this huge function.
    def _get_params(self) -> dict[str, Any]:
        """Get a dictionary of parameters from the .par file for use in
         constructing the spectrum.

        Returns:
            Dictionary of params. Must always have keys 'ndim', 'header_size',
             'shape', 'block_shape', 'nuclei', 'ref_ppm', 'ref_coord',
             'spectrometer_frequency', 'spectral_width'.
             May also have keys 'integer', 'swap', 'endianness'.

        Raises:
            ValueError: if unsupported keys 'varian', 'blocks', 'sigmas',
             'params' are detected. .par files containing these keys are
             intended for internal use within Azara only.
        """
        raw_params = _parse_par_file(self.par_path)

        params = {"header_size": 0, "dtype": np.float32}

        def unsupported_key_error(key):
            return ValueError(
                f"""Unsupported key '{key}' found in {self.par_path}.
                 Azara data must be blocked."""
            )

        # Iterate over all params to get global params
        for param in raw_params:
            key = param[0]
            value = param[1] if len(param) >= 2 else None

            match key:
                case "ndim":
                    params["ndim"] = int(value)
                case "file":
                    params["data_file"] = value.strip()
                case "head":
                    params["header_size"] = int(value)
                case "int":
                    params["integer"] = True
                    params["dtype"] = np.int32
                    warn(
                        f"""Key 'int' found in {
                            self.par_path
                        }. Attempting to interpret data as integer."""
                    )
                case "big_endian":
                    params["endianness"] = False
                    params["dtype"] = np.dtype(">f4")
                case "little_endian":
                    params["endianness"] = True
                    params["dtype"] = np.float32
                case "swap" | "deflate" | "reflate":
                    params[key] = value
                    warn(
                        f"""Ignoring parameter '{key}' in Azara params file {
                            self.par_path
                        }"""
                    )
                case "varian":
                    params["varian"] = value
                    raise unsupported_key_error("varian")
                case "blocks":
                    params["blocks"] = value
                    raise unsupported_key_error("blocks")

        # Record where 'dim' keywords are found, as delimiters of blocks of
        # axis params, including one after the final line.
        dim_positions = [i for i, param in enumerate(raw_params) if param[0] == "dim"]
        dim_positions.append(len(raw_params))

        ndim = params["ndim"]

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
            for param in raw_params[start:stop]:
                key = param[0]
                value = param[1] if len(param) >= 2 else None

                match key:
                    case "dim":
                        dim = int(value)
                    case "npts":
                        shape[dim - 1] = int(value)
                    case "block":
                        block_shape[dim - 1] = int(value)
                    case "sw":
                        spectral_width[dim - 1] = float(value)
                    case "sf":
                        spectrometer_frequency[dim - 1] = float(value)
                    case "refppm":
                        ref_ppm[dim - 1] = float(value)
                    case "refpt":
                        ref_coord[dim - 1] = float(value)
                    case "nuc":
                        nuclei[dim - 1] = value.strip()
                    case "params":
                        raise unsupported_key_error("params")
                    case "sigmas":
                        raise unsupported_key_error("sigmas")

        # Invert to row-major order
        params["shape"] = tuple(shape[::-1])
        params["block_shape"] = tuple(block_shape[::-1])
        params["nuclei"] = tuple(nuclei[::-1])
        params["spectral_width"] = tuple(spectral_width[::-1])
        params["spectrometer_frequency"] = tuple(spectrometer_frequency[::-1])
        params["ref_ppm"] = tuple(ref_ppm[::-1])
        params["ref_coord"] = tuple(ref_coord[::-1])

        return params

    def get_ndim(self) -> int:
        return self._params["ndim"]

    def get_nuclei(self) -> tuple[NMRNucleus, ...]:
        return self._params["nuclei"]

    def get_data(self) -> BlockedSpectrumDataSource:
        return BlockedSpectrumDataSource(
            path=self.spc_path,
            shape=self._params["shape"],
            block_shape=self._params["block_shape"],
            header_size=self._params["header_size"],
            dtype=self._params["dtype"],
        )

    def get_transform(self) -> SpectrumTransform:
        return SpectrumTransform.from_reference(
            shape=self._params["shape"],
            spectral_width=self._params["spectral_width"],
            spectrometer_frequency=self._params["spectrometer_frequency"],
            ref_coord=self._params["ref_coord"],
            ref_shift=self._params["ref_ppm"],
        )

    @classmethod
    def can_read(cls, path: Path) -> bool:
        return path.is_file() and path.suffix in [".spc", ".par"]
