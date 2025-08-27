import numpy as np
from matplotlib import pyplot as plt
from bioshift.core import Spectrum, NMRNucleus


def _axis_label(nuc: NMRNucleus) -> str:
    return f"{nuc} [ppm]"


def plot_spectrum_heatmap(spectrum: Spectrum, ax=None, **kwargs):
    if spectrum.ndim != 2:
        raise ValueError("Spectrum must be 2D for heatmap plot.")

    if ax is None:
        fig, ax = plt.subplots()

    transform = spectrum.transform
    ny, nx = transform.shape

    min_shift = transform.grid_to_shift(np.array([0, 0]))
    max_shift = transform.grid_to_shift(np.array(list(transform.shape)))

    x = np.linspace(min_shift[1], max_shift[1], nx)
    y = np.linspace(min_shift[0], max_shift[0], ny)
    X, Y = np.meshgrid(x, y)

    intensity = spectrum.array[::, ::-1]

    ax.imshow(
        intensity,
        extent=[min_shift[1], max_shift[1], min_shift[0], max_shift[0]],
        aspect="auto",
        **kwargs,
    )

    ax.invert_xaxis()
    ax.invert_yaxis()

    ax.set_xlabel(_axis_label(spectrum.nuclei[1]))
    ax.set_ylabel(_axis_label(spectrum.nuclei[0]))

    return ax


def plot_spectrum_contour(
    spectrum: Spectrum, threshold=500, ax=None, levels=25, linewidths=0.65, **kwargs
):
    if spectrum.ndim != 2:
        raise ValueError("Spectrum must be 2D for contour plot.")

    if ax is None:
        fig, ax = plt.subplots()

    transform = spectrum.transform
    ny, nx = transform.shape

    min_shift = transform.grid_to_shift(np.array([0, 0]))
    max_shift = transform.grid_to_shift(np.array(list(transform.shape)))

    x = np.linspace(min_shift[1], max_shift[1], nx)
    y = np.linspace(min_shift[0], max_shift[0], ny)
    X, Y = np.meshgrid(x, y)

    intensity = spectrum.array[::-1, ::-1]
    intensity = np.where(np.abs(intensity) < threshold, 0, intensity)

    positive_contours = []
    max = np.max(np.abs(intensity))

    # base = 1.5
    # start = np.log(min_contour) / np.log(base)
    # stop = np.log2(max) / np.log(base)
    # positive_contours = np.logspace(start, stop, num=n_levels//2, base=base)

    if isinstance(levels, int):
        positive_contours = np.linspace(0, max, num=levels // 2)
        positive_contours = np.array(
            [level for level in positive_contours if level >= threshold]
        )
        negative_contours = -positive_contours[::-1]

        levels = np.concatenate((negative_contours, positive_contours))

    ax.contour(X, Y, intensity, levels=levels, linewidths=linewidths, **kwargs)

    ax.invert_xaxis()
    ax.invert_yaxis()

    ax.set_xlabel(_axis_label(spectrum.nuclei[1]))
    ax.set_ylabel(_axis_label(spectrum.nuclei[0]))

    return ax


def plot_spectrum_line(spectrum: Spectrum, ax=None, **kwargs):
    if spectrum.ndim != 1:
        raise ValueError("Spectrum must be 1D for line plot.")

    if ax is None:
        fig, ax = plt.subplots()

    transform = spectrum.transform
    nx = transform.shape[0]
    min_shift = transform.grid_to_shift(np.array([0, 0]))
    max_shift = transform.grid_to_shift(np.array(list(transform.shape)))

    x = np.linspace(min_shift, max_shift, nx)

    intensity = spectrum.array[::-1]

    ax.plot(x, intensity, **kwargs)

    ax.invert_xaxis()
    ax.set_xlabel(_axis_label(spectrum.nuclei[0]))

    return ax
