from bioshift.core.spectrum import Spectrum
from bioshift.core.peak import Peak, Shift, peak_list_from_array

import numpy as np
from numpy.typing import NDArray
import skimage


def difference_of_gaussians(
    spectrum: Spectrum,
    min_sigma,
    max_sigma,
    sigma_ratio=1.6,
    threshold=0.5,
    overlap=0.5,
    threshold_rel=None,
    exclude_border=False,
) -> NDArray:

    min_sigma_scaled = np.array(min_sigma) * spectrum.transform.inverse_scaling
    max_sigma_scaled = np.array(max_sigma) * spectrum.transform.inverse_scaling

    # skimage is expecting an image, so need to invert the coordinates
    arr = np.flip(spectrum.array)

    feature_array = skimage.feature.blob_dog(
        image=arr,
        min_sigma=min_sigma_scaled,
        max_sigma=max_sigma_scaled,
        sigma_ratio=sigma_ratio,
        threshold=threshold,
        overlap=overlap,
        threshold_rel=threshold_rel,
        exclude_border=exclude_border,
    )

    shifts = spectrum.transform.grid_to_shift(feature_array[:, :2])
    linewidths = feature_array[:, 2:] * spectrum.transform.scaling

    return peak_list_from_array(
        shifts=shifts, linewidths=linewidths, nuclei=spectrum.nuclei
    )

    # return np.stack(shifts, linewidths, axis=2)
