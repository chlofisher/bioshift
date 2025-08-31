from bioshift.core import Spectrum
from bioshift.visualization import plot_spectrum_heatmap, plot_spectrum_contour
from matplotlib import pyplot as plt
import matplotlib
import numpy as np


class SliceViewer:
    spectrum: Spectrum
    ax: matplotlib.axis.Axis
    step: float
    z: float
    z_min: float
    z_max: float
    slice_axis: int

    def __init__(self, spectrum, slice_axis=0, z0=None, step=None, **kwargs):
        if spectrum.ndim != 3:
            raise ValueError("Spectrum must be 3D for slice plot.")

        fig, ax = plt.subplots()

        transform = spectrum.transform
        z_min = transform.bounds[0][slice_axis]
        z_max = transform.bounds[1][slice_axis]

        if z0 is None:
            z0 = z_min

        if step is None:
            step = transform.scaling[slice_axis]

        self.spectrum = spectrum
        self.ax = ax
        self.fig = fig
        self.slice_axis = slice_axis
        self.z = z0
        self.z_min = z_min
        self.z_max = z_max
        self.step = step
        self.kwargs = kwargs

        self.fig.canvas.mpl_connect("key_press_event", self._process_key)

    def _get_slice(self) -> Spectrum:
        slice = self.spectrum.slice(axis=self.slice_axis, z=self.z)
        return slice

    def _process_key(self, event):
        fig = event.canvas.figure
        if event.key == "up":
            self._previous_slice()
        elif event.key == "down":
            self._next_slice()
        fig.canvas.draw()

    def _previous_slice(self):
        self.z = np.clip(self.z - self.step, self.z_min, self.z_max)
        self._update()

    def _next_slice(self):
        self.z = np.clip(self.z + self.step, self.z_min, self.z_max)
        self._update()


class HeatmapSliceViewer(SliceViewer):
    def __init__(self, spectrum, slice_axis=0, z0=None, step=None, norm=None, **kwargs):
        max = np.max(np.abs(spectrum.array))
        if norm is None:
            norm = matplotlib.colors.CenteredNorm(vcenter=0, halfrange=max * 0.5)

        self.norm = norm

        super().__init__(spectrum, slice_axis, z0, step, **kwargs)

    def _update(self):
        slice = self._get_slice()
        self.ax.images[0].set_array(slice.array[::, ::-1])

    def plot(self):
        slice: Spectrum = self._get_slice()
        plot_spectrum_heatmap(slice, ax=self.ax, norm=self.norm, **self.kwargs)

        return self.ax


class ContourSliceViewer(SliceViewer):
    def __init__(
        self, spectrum, threshold, slice_axis=0, z0=None, step=None, norm=None, **kwargs
    ):
        self.threshold = threshold

        super().__init__(spectrum, slice_axis, z0, step, **kwargs)

    def _update(self):
        slice = self._get_slice()

        for collection in self.ax.collections:
            collection.remove()

        plot_spectrum_contour(
            slice,
            threshold=self.threshold,
            ax=self.ax,
            invert_axes=False,
            **self.kwargs,
        )

    def plot(self):
        slice: Spectrum = self._get_slice()
        plot_spectrum_contour(
            slice, threshold=self.threshold, ax=self.ax, invert_axes=True, **self.kwargs
        )

        return self.ax
