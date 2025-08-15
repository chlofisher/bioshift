import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
import csv


@dataclass
class Peak:
    position: NDArray
    width: NDArray


@dataclass
class PeakList:
    positions: NDArray
    widths: NDArray

    def __iter__(self):
        return iter(Peak(pos, width) for pos, width in zip(self.positions, self.widths))

    def __len__(self):
        return self.positions.shape[0]

    def __getitem__(self, key: slice | int):
        peaks = [
            Peak(pos, width)
            for pos, width in zip(self.positions[key, :], self.widths[key, :])
        ]

        if isinstance(key, slice):
            return peaks
        elif isinstance(key, int):
            return Peak(self.positions[key, :], self.widths[key, :])

    def write_csv(self, path):
        ndim = self.positions.shape[1]
        with open(path, "w") as file:
            fieldnames = (
                [f"shift[{i}]" for i in range(ndim)]
                + [f"width[{i}]" for i in range(ndim)]
            )
            
            writer = csv.writer(file)
            writer.writerow(fieldnames)
            for peak in self:
                row = peak.position.tolist() + peak.width.tolist()
                writer.writerow(row)

    def from_csv(path):
        with open(path, "r") as file:
            reader = csv.reader(file)
            header = next(reader)

            positions = []
            widths = []
            for row in reader:
                row = [float(val) for val in row]
                ndim = len(row) // 2
                positions.append(row[:ndim])
                widths.append(row[ndim:])

        return PeakList(
            positions=np.array(positions),
            widths=np.array(widths)
        )
