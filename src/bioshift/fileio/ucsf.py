from __future__ import annotations
from pathlib import Path
import struct
import numpy as np

from bioshift.core.spectrum import NMRNucleus
from bioshift.core.spectrumreference import SpectrumReference
from bioshift.fileio.blockedspectrum import BlockedSpectrumDataSource
from bioshift.fileio.spectrumreader import SpectrumReader


GLOBAL_HEADER_SIZE = 180
AXIS_HEADER_SIZE = 128


class UCSFSpectrumReader(SpectrumReader):

    path: Path
    params: dict

    def __init__(self, path: Path):
        self.path = path
        self.params = self.get_params()

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

        params = {}

        with open(self.path, 'rb') as file:
            file.seek(10)
            # Get number of dimensions first to determine header size
            ndim: int = int.from_bytes(file.read(1))
            params['ndim'] = ndim

            file.seek(0)
            header_bytes: bytes = file.read(self.get_header_size(ndim))

        global_header = struct.unpack('>10sBBxB', header_bytes[:14])

        params['file_type'] = global_header[0].rstrip(b'\x00').decode('ascii')
        # Ignore global_header[1] as this is the ndim we found earlier
        params['n_components'] = int(global_header[2])
        params['version'] = int(global_header[3])

        shape = [None] * ndim
        block_shape = [None] * ndim
        nuclei = [None] * ndim
        spectrometer_frequency = [None] * ndim
        spectral_width = [None] * ndim
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
            spectral_width[i] = float(axis_header[4])
            center_shift[i] = float(axis_header[5])

        params['shape'] = tuple(shape)
        params['block_shape'] = tuple(block_shape)
        params['nuclei'] = tuple(nuclei)

        params['spectrometer_frequency'] = tuple(spectrometer_frequency)
        params['spectral_width'] = tuple(spectral_width)
        params['ref_ppm'] = tuple(center_shift)
        params['ref_coord'] = tuple(n/2 for n in shape)

        return params

    def get_ndim(self) -> int:
        return self.params['ndim']

    def get_nuclei(self) -> tuple[NMRNucleus, ...]:
        return self.params['nuclei']

    def get_reference(self) -> SpectrumReference:
        return SpectrumReference(
            spectrum_shape=self.params['shape'],
            spectral_width=self.params['spectral_width'],
            spectrometer_frequency=self.params['spectrometer_frequency'],
            ref_coord=self.params['ref_coord'],
            ref_ppm=self.params['ref_ppm']
        )

    def get_data(self) -> BlockedSpectrumDataSource:
        """Create a BlockedSpectrumDataSource object for the new spectrum.

        Returns:
            UCSFDataSource object.
        """
        return BlockedSpectrumDataSource(
            path=self.path,
            shape=self.params['shape'],
            block_shape=self.params['block_shape'],
            header_size=self.get_header_size(self.params['ndim']),
            dtype=np.dtype('>f4')
        )

    def get_header_size(self, ndim):
        """Calculates the size of the header in the .ucsf binary file from the 
        number of dimensions in the spectrum.

        Returns:
            Size of the header in bytes.
        """
        return GLOBAL_HEADER_SIZE + AXIS_HEADER_SIZE * ndim

    @classmethod
    def can_read(cls, path: Path) -> bool:
        return path.suffix == '.ucsf'
