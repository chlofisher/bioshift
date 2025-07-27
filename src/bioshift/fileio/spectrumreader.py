from abc import ABC, abstractmethod
from pathlib import Path

from bioshift.core.spectrum import Spectrum, NMRNucleus
from bioshift.core.spectrumreference import SpectrumReference
from bioshift.fileio.spectrumdatasource import SpectrumDataSource


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
        ndim: int = self.get_ndim()
        nuclei: tuple[NMRNucleus, ...] = self.get_nuclei()
        datasource: SpectrumDataSource = self.get_data()
        reference: SpectrumReference = self.get_reference()

        return Spectrum(
            ndim=ndim,
            nuclei=nuclei,
            data_source=datasource,
            transform=reference.transform()
        )

    @abstractmethod
    def get_ndim(self) -> int:
        ...

    @abstractmethod
    def get_nuclei(self) -> tuple[NMRNucleus, ...]:
        ...

    @abstractmethod
    def get_data(self) -> SpectrumDataSource:
        ...

    @abstractmethod
    def get_reference(self) -> SpectrumReference:
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
