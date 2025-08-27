from bioshift.core import Spectrum
from bioshift.visualization import plot_spectrum_heatmap
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

    _slice_cache: dict[float, Spectrum]

    def __init__(
        self,
        spectrum,
        slice_axis=0,
        z0=None,
        step=None,
    ):
        fig, ax = plt.subplots()

        transform = spectrum.transform
        z_min = transform.bounds[0][slice_axis]
        z_max = transform.bounds[1][slice_axis]

        if z0 is None:
            z0 = (z_min + z_max) / 2

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
        self._slice_cache = {}

    def _get_slice(self) -> Spectrum:
        if self.z in self._slice_cache:
            return self._slice_cache[self.z]

        slice = self.spectrum.slice(axis=self.slice_axis, z=self.z)
        self._slice_cache[self.z] = slice

        return slice

    def _process_key(self, event):
        fig = event.canvas.figure
        if event.key == '1':
            self._previous_slice()
        elif event.key == '2':
            self._next_slice()
        fig.canvas.draw()

    def _previous_slice(self):
        self.z = np.clip(self.z - self.step, self.z_min, self.z_max)
        self._update()

    def _next_slice(self):
        self.z = np.clip(self.z + self.step, self.z_min, self.z_max)
        self._update()

    def _update(self):
        slice = self._get_slice()
        self.ax.images[0].set_array(slice.array[::, ::-1])

    def plot(self):
        slice: Spectrum = self._get_slice()
        plot_spectrum_heatmap(slice, ax=self.ax)
        self.fig.canvas.mpl_connect('key_press_event', self._process_key)
