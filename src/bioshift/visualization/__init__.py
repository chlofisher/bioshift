try:
    from matplotlib import pyplot as _
except ImportError as e:
    raise ImportError(
        "The visualization module requires matplotlib (optional dependency). "
        "Install it using `pip install bioshift[plot]`."
    ) from e

from .plot import (
    plot_spectrum_heatmap,
    plot_spectrum_contour,
    plot_spectrum_line,
)

from .sliceviewer import SliceViewer, HeatmapSliceViewer, ContourSliceViewer
