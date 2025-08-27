from os import PathLike
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Self
import csv


@dataclass
class Peak:
    """
    Class to store the chemical shift and width of a peak.
    """

    shift: NDArray
    width: NDArray


@dataclass
class PeakList:
    """
    Stores an array of shifts and widths to represent a set of peaks.
    PeakLists can be iterated over or indexed as if they were a collection of Peak objects.

    Usage:
    ```python
    shifts = np.array([
        [0.6, 2.5, 1.6],
        [15.6, 20.1, 8.5]
    ])
    widths = np.array([
            [1.2, 0.8, 1.],
            [2.5, 2.8, 2.9]
    ])

    peaklist = PeakList(shifts, widths)

    print(len(peaklist))
    # output: 2

    print(peak[0])
    # output:
    #   Peak(
    #       shift=array([0.6, 2.5, 1.6]),
    #       width=array([1.2, 0.8, 1. ])
    #   )

    for peak in peaklist:
        print(peak)

    # output:
    #   Peak(
    #       shift=array([0.6, 2.5, 1.6]),
    #       width=array([1.2, 0.8, 1. ])
    #   )
    #   Peak(
    #       shift=array([15.6, 20.1,  8.5]),
    #       width=array([2.5, 2.8, 2.9])
    #   )
    ```
    """

    shifts: NDArray
    widths: NDArray | None

    def __iter__(self):
        if self.widths is not None:
            return iter(
                Peak(shift, width) for shift, width in zip(self.shifts, self.widths)
            )
        else:
            return iter(Peak(shift, None) for shift in self.shifts)

    def __len__(self):
        return self.shifts.shape[0]

    def __getitem__(self, key: slice | int):
        peaks = [
            Peak(pos, width)
            for pos, width in zip(self.shifts[key, :], self.widths[key, :])
        ]

        if isinstance(key, slice):
            return peaks
        elif isinstance(key, int):
            return Peak(self.shifts[key, :], self.widths[key, :])

    def write_csv(self, path: str | PathLike):
        """
        Save the peak list to a CSV file at the given path.

        Args:
            path: Path to the destination file.
        """
        ndim = self.shifts.shape[1]
        with open(path, "w") as file:
            fieldnames = [f"shift[{i}]" for i in range(ndim)] + [
                f"width[{i}]" for i in range(ndim)
            ]

            writer = csv.writer(file)
            writer.writerow(fieldnames)
            for peak in self:
                row = peak.shift.tolist() + peak.width.tolist()
                writer.writerow(row)

    def from_csv(path) -> Self:
        """
        Read a peak list from a CSV file.

        Returns:
            PeakList object.
        """
        with open(path, "r") as file:
            reader = csv.reader(file)
            header = next(reader)

            shifts = []
            widths = []
            for row in reader:
                row = [float(val) for val in row]
                ndim = len(row) // 2
                shifts.append(row[:ndim])
                widths.append(row[ndim:])

        return PeakList(shifts=np.array(shifts), widths=np.array(widths))
