from __future__ import annotations
from pathlib import Path

from protnmr.fileio.spectrumreader import SpectrumReader
from protnmr.fileio.spectrumdatasource import SpectrumDataSource
from protnmr.core.spectrumparams import SpectrumParams


class NMRPipeSpectrumReader(SpectrumReader):
    def __init__(self, path: Path):
        self.path = path

    @classmethod
    def from_path(cls, path: Path) -> NMRPipeSpectrumReader:
        return cls(path)

    def get_params(self) -> SpectrumParams:
        raise NotImplementedError()

    def get_data(self) -> SpectrumDataSource:
        raise NotImplementedError()

    @classmethod
    def can_read(cls, path: Path) -> bool:
        return False
