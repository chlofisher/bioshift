import numpy as np
from numpy.typing import NDArray
import skimage

from bioshift.spectra import Spectrum


def difference_of_gaussians(
    spectrum: Spectrum,
    sigma: NDArray,
    k: float,
    threshold_rel: float = 0.1,
    negative_peaks: bool = True,
):
    sigma_scaled = np.abs(np.array(sigma) * spectrum.transform.inverse_scaling)

    normalized = spectrum.normalize()

    dog = skimage.filters.difference_of_gaussians(
        image=normalized.array,
        low_sigma=sigma_scaled,
        high_sigma=sigma_scaled * k,
    )

    positive_features = skimage.feature.peak_local_max(dog, threshold_rel=threshold_rel)

    if negative_peaks:
        negative_features = skimage.feature.peak_local_max(
            dog, threshold_rel=threshold_rel
        )

        features = np.vstack((positive_features, negative_features))
    else:
        features = positive_features

    shifts = spectrum.transform.grid_to_shift(features)

    return shifts
