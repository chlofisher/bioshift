from __future__ import annotations
from typing import Any
from pathlib import Path
import struct
import numpy as np

from bioshift.core.spectrum import NMRNucleus
from bioshift.core.spectrumtransform import SpectrumTransform
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
        return cls(path)

    def get_params(self) -> dict[str, Any]:
        """Read the spectrum metadata from the .ucsf file header.

        Returns:
            Dictionary of params. Must always have keys 'ndim', 'header_size',
            'shape', 'block_shape', 'nuclei', 'ref_ppm', 'ref_coord',
            'spectrometer_frequency', 'spectral_width'.
            May also have keys 'integer', 'swap', 'endianness'.
        """

        params = {}

        with open(self.path, "rb") as file:
            file.seek(10)
            # Get number of dimensions first to determine header size
            ndim: int = int.from_bytes(file.read(1))
            params["ndim"] = ndim

            file.seek(0)
            header_size = self.get_header_size(ndim)
            params["header_size"] = header_size
            header_bytes: bytes = file.read(header_size)

        global_header = struct.unpack(">10sBBxB", header_bytes[:14])

        file_type = global_header[0].rstrip(b"\x00").decode("ascii")
        if file_type != "UCSF NMR":
            raise ValueError(
                f"""Invalid file type '{file_type}' in
                {self.path} header. Expecting 'UCSF NMR.'"""
            )

        # Ignore global_header[1] as this is the ndim we found earlier

        n_components = int(global_header[2])
        if n_components != 1:
            raise ValueError(
                f"""Invalid number of components {n_components}
                found in {self.path} header. Only real data is supported."""
            )

        version = int(global_header[3])
        if version != 2:
            raise ValueError(
                f"""Invalid format version {version} found in {self.path}.
                Expecting version 2."""
            )

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
            axis_header = struct.unpack(">6s2xi4xifff", axis_header_bytes[:32])

            nuclei[i] = axis_header[0].rstrip(b"\x00").decode("ascii")
            shape[i] = int(axis_header[1])
            block_shape[i] = int(axis_header[2])
            spectrometer_frequency[i] = float(axis_header[3])
            spectral_width[i] = float(axis_header[4])
            center_shift[i] = float(axis_header[5])

        params["shape"] = tuple(shape)
        params["block_shape"] = tuple(block_shape)
        params["nuclei"] = tuple(nuclei)

        params["spectrometer_frequency"] = tuple(spectrometer_frequency)
        params["spectral_width"] = tuple(spectral_width)
        params["ref_ppm"] = tuple(center_shift)
        params["ref_coord"] = tuple(n / 2 for n in shape)

        return params

    def get_ndim(self) -> int:
        return self.params["ndim"]

    def get_nuclei(self) -> tuple[NMRNucleus, ...]:
        return self.params["nuclei"]

    def get_transform(self) -> SpectrumTransform:
        return SpectrumTransform.from_reference(
            shape=self.params["shape"],
            spectral_width=self.params["spectral_width"],
            spectrometer_frequency=self.params["spectrometer_frequency"],
            ref_coord=self.params["ref_coord"],
            ref_shift=self.params["ref_ppm"],
        )

    def get_data(self) -> BlockedSpectrumDataSource:
        return BlockedSpectrumDataSource(
            path=self.path,
            shape=self.params["shape"],
            block_shape=self.params["block_shape"],
            header_size=self.params["header_size"],
            dtype=np.dtype(">f4"),
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
        return path.suffix == ".ucsf"
