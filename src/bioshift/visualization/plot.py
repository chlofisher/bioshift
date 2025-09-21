import matplotlib
from matplotlib.colors import CenteredNorm
from matplotlib import pyplot as plt
import numpy as np
from bioshift.core import Spectrum, NMRNucleus


def _axis_label(nuc: NMRNucleus) -> str:
    return f"{str(nuc)} [ppm]"


def _init_axes(ax):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    return fig, ax


def heatmap(
    spectrum: Spectrum,
    norm=CenteredNorm(0),
    ax=None,
    aspect="auto",
    show=False,
    **kwargs,
):
    fig, ax = _init_axes(ax)

    if spectrum.ndim != 2:
        raise ValueError("Spectrum must be 2D for heatmap plot.")

    transform = spectrum.transform
    ny, nx = transform.shape

    min_shift = transform.grid_to_shift(np.array([0, 0]))
    max_shift = transform.grid_to_shift(np.array(list(transform.shape)))

    x = np.linspace(min_shift[1], max_shift[1], nx)
    y = np.linspace(min_shift[0], max_shift[0], ny)
    X, Y = np.meshgrid(x, y)

    intensity = spectrum.array[::, ::-1]

    im = ax.imshow(
        intensity,
        extent=[min_shift[1], max_shift[1], min_shift[0], max_shift[0]],
        aspect=aspect,
        norm=norm,
        **kwargs,
    )

    ax.invert_xaxis()
    ax.invert_yaxis()

    ax.set_xlabel(_axis_label(spectrum.nuclei[1]))
    ax.set_ylabel(_axis_label(spectrum.nuclei[0]))

    fig.colorbar(im)

    if show:
        plt.show()

    return ax


def contour(
    spectrum: Spectrum,
    threshold,
    ax=None,
    levels=25,
    linewidths=0.65,
    show=False,
    **kwargs,
):
    if spectrum.ndim != 2:
        raise ValueError("Spectrum must be 2D for contour plot.")

    fig, ax = _init_axes(ax)

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

    if show:
        plt.show()

    return ax


def line(spectrum: Spectrum, ax=None, show=False, **kwargs):
    if spectrum.ndim != 1:
        raise ValueError("Spectrum must be 1D for line plot.")

    fig, ax = _init_axes(ax)

    transform = spectrum.transform
    nx = transform.shape[0]
    min_shift = transform.grid_to_shift(np.array([0, 0]))
    max_shift = transform.grid_to_shift(np.array(list(transform.shape)))

    x = np.linspace(min_shift, max_shift, nx)

    intensity = spectrum.array[::-1]

    ax.plot(x, intensity, **kwargs)

    ax.invert_xaxis()
    ax.set_xlabel(_axis_label(spectrum.nuclei[0]))

    if show:
        plt.show()

    return ax
