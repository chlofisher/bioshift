import numpy as np
from numpy.typing import NDArray


def _peak_distance_matrix(
    nhsqc_peaks: NDArray, hncacb_peaks: NDArray, axis: int, scaling: NDArray
):
    num_nhsqc_peaks = nhsqc_peaks.shape[0]
    num_hncacb_peaks = hncacb_peaks.shape[0]

    projected_hncacb_peaks = np.delete(hncacb_peaks, obj=axis, axis=1)

    # (num_hsqc_peaks, num_hncacb_peaks, ndim)
    nhsqc_matrix = np.repeat(nhsqc_peaks[:, None, :], num_hncacb_peaks, axis=1)
    hncacb_matrix = np.repeat(
        projected_hncacb_peaks[None, :, :], num_nhsqc_peaks, axis=0
    )

    distance_matrix = np.sum(((nhsqc_matrix - hncacb_matrix) * scaling) ** 2, axis=2)

    return distance_matrix


def match_projected_peaks(
    nhsqc_peaks: NDArray, hncacb_peaks: NDArray, axis: int, scaling: NDArray
):
    distance_matrix = _peak_distance_matrix(nhsqc_peaks, hncacb_peaks, axis, scaling)
    indices = np.argmin(distance_matrix, axis=0)

    return indices
