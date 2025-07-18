from pathlib import Path
from os import PathLike

from bioshift.core.spectrum import Spectrum
from bioshift.fileio.azara import AzaraSpectrumReader
from bioshift.fileio.ucsf import UCSFSpectrumReader
from bioshift.fileio.nmrpipe import NMRPipeSpectrumReader


REGISTRY = [
    AzaraSpectrumReader,
    UCSFSpectrumReader,
    NMRPipeSpectrumReader
]


def load_spectrum(path: str | PathLike) -> Spectrum:
    """Helper function to dynamically dispatch a concrete SpectrumReader to 
    read a spectrum from a path.

    Args:
        path: Path to spectrum on disk.

    Returns:
        Spectrum loaded from the path.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f'Path {path} does not exist.')

    for reader_cls in REGISTRY:
        if reader_cls.can_read(path):
            reader = reader_cls.from_path(path)
            return reader.read()
