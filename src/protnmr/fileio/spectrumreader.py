from abc import ABC, abstractmethod
from pathlib import Path

from protnmr.core.spectrum import Spectrum
from protnmr.core.spectrumparams import SpectrumParams
from protnmr.fileio.spectrumdatasource import SpectrumDataSource


class SpectrumReader(ABC):
    """Base class for spectrum readers. Specifies an interface for producing a 
    SpectrumParams and a SpectrumDataSource from different file formats 
    implemented by concrete SpectrumReaders.
    """

    @abstractmethod
    def __init__(self, path: Path):
        ...

    @classmethod
    @abstractmethod
    def from_path(cls, path: Path):
        ...

    def read(self) -> Spectrum:
        """Creates a new spectrum from a SpectrumParams and SpectrumDataSource
        constructed by reading from self.path.

        Returns:
            New spectrum object.
        """
        params = self.get_params()
        datasource = self.get_data()

        return Spectrum(params, datasource)

    @abstractmethod
    def get_params(self) -> SpectrumParams:
        """Creates a SpectrumParams object for the spectrum.

        Returns:
            SpectrumParams object containing metadata read from path.
        """
        ...

    @abstractmethod
    def get_data(self) -> SpectrumDataSource:
        """Creates a SpectrumDataSource object for the spectrum.

        Returns:
            SpectrumDataSource object which reads raw spectrum array from disk.
        """
        ...

    @classmethod
    @abstractmethod
    def can_read(cls, path: Path) -> bool:
        """Specifies whether or not a given path can be read by a particular 
        concrete SpectrumReader implementation. Used to dynamically determine
        which concrete SpectrumReader to use to read from a given path.

        Returns:
            True if the given path can be read by the concrete SpectrumReader 
            class.
        """
        ...
