from pathlib import Path
from os import PathLike

from bioshift.core.spectrum import Spectrum
from bioshift.fileio.azara import AzaraSpectrumReader
from bioshift.fileio.ucsf import UCSFSpectrumReader


REGISTRY = [AzaraSpectrumReader, UCSFSpectrumReader]


def load_spectrum(path: str | PathLike) -> Spectrum:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Path {path} does not exist.")

    for reader_cls in REGISTRY:
        if reader_cls.can_read(path):
            reader = reader_cls.from_path(path)
            return reader.read()
