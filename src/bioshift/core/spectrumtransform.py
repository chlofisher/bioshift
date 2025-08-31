import numpy as np
from typing import Self
from numpy.typing import NDArray


class SpectrumTransform:
    """Represents a diagonal affine transformation used to map between
    coordinates in the raw spectrum array and chemical shift values.
    """

    ndim: int
    """Number of dimensions in the spectrum."""

    shape: NDArray
    """Shape of the underlying data array."""

    bounds: NDArray

    scaling: NDArray
    """Vector of diagonal components of the affine transformation matrix. 
    Components are scaling values along each axis."""
    offset: NDArray
    """Constant offset vector for the affine transformation."""

    inverse_scaling: NDArray
    inverse_offset: NDArray

    def __init__(self, ndim, shape, scaling, offset):
        """Initialise the SpectrumTransform. Performs validation logic and
        copies the input arrays to ensure immutability.

        Raises:
            ValueError: If any of the scaling values are zero,
                to enforce that the transformation matrix is non-singular.
            ValueError: If the shape of the scaling and the offset vectors do
                not match.
        """
        if scaling.ndim != 1:
            raise ValueError()

        if offset.ndim != 1:
            raise ValueError()

        if len(scaling) != ndim:
            raise ValueError()

        if len(offset) != ndim:
            raise ValueError()

        if not all(x != 0 for x in scaling):
            raise ValueError(
                """Non-invertible transformation: 
                SpectrumTransform scaling values must all be non-zero.
                """.strip()
            )

        self.ndim = ndim
        self.shape = shape
        self.scaling = np.array(scaling, dtype=np.float32).copy()
        self.offset = np.array(offset, dtype=np.float32).copy()

    @property
    def bounds(self) -> NDArray:
        """Bounds of the spectrum (in array grid coordinates)."""
        array_bounds = np.vstack((np.zeros(self.ndim), self.shape))

        return self.grid_to_shift(array_bounds)

    @property
    def inverse_scaling(self) -> NDArray:
        """Vector containing the diagonal entries of the affine transformation matrix for
        the inverse transform. Used to convert from chemical shifts to grid coords."""
        return 1 / self.scaling

    @property
    def inverse_offset(self) -> NDArray:
        """Vector offset for the inverse transform.
        Used to convert from chemical shifts to grid coords."""
        return -self.offset / self.scaling

    def grid_to_shift(self, x: NDArray) -> NDArray:
        """
        Convert a grid coord to a chemical shift.

        Args:
            x:
                Array containing grid coordinates of points in the spectrum.
                Must be broadcastable to (ndim,).
        Returns:
            Array of corresponding chemical shift values
        """
        return x * self.scaling + self.offset

    def shift_to_grid(self, x) -> NDArray:
        """Convert a grid coord to a chemical shift.
        Args:
            x: Array containing chemical shift coordinates. Must be broadcastable to (ndim,).
        Returns:
            Array of corresponding grid coordinates.
        """
        return x * self.inverse_scaling + self.inverse_offset

    @classmethod
    def from_reference(
        cls,
        shape: NDArray,
        spectral_width: NDArray,
        spectrometer_frequency: NDArray,
        ref_coord: NDArray,
        ref_shift: NDArray,
    ) -> Self:
        """
        Create a SpectrumTransform from spectrum referencing information.
        All arguments must be NDArrays with shape `(ndim,)`

        Args:
            shape :
                Number of data points along each axis of the spectrum.
            spectral_width:
                Difference in chemical shift across the width of the spectrum along each axis,
                measured in Hz.
            spectrometer_frequency:
                Frequency of spectrometer along each axis, measured in MHz.
                Required to convert spectral width into ppm.
            ref_coord:
                Grid coordinates of a reference point in the spectrum.
            ref_shift:
                Chemical shift vector of the point located at the reference coordinate.
        Returns:
            `SpectrumTransform` object with scaling and offset vectors calculated
            from the reference data.
        """

        w = np.array(spectral_width)
        N = np.array(shape)
        f = np.array(spectrometer_frequency)

        delta_0 = np.array(ref_shift)
        i_0 = np.array(ref_coord)

        scaling = w / (N * f)
        offset = delta_0 - scaling * i_0

        return cls(ndim=len(shape), shape=shape, scaling=scaling, offset=offset)

    def slice(self, axis: int) -> Self:
        new_shape = tuple(n for i, n in enumerate(self.shape) if i != axis)
        new_scaling = np.delete(self.scaling, axis)
        new_offset = np.delete(self.offset, axis)

        return SpectrumTransform(
            ndim=self.ndim - 1, shape=new_shape, scaling=new_scaling, offset=new_offset
        )
