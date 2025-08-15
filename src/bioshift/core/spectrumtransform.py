import numpy as np
from numpy.typing import NDArray


class SpectrumTransform:
    """Represents a diagonal affine transformation used to map between
    coordinates in the raw spectrum array and chemical shift values.

    Attributes:
        scaling: Vector of scaling factors for each axis.
        offset: Offset vector.

    Properties:
        inverse: The inverse of the transform.
    """

    ndim: int
    shape: NDArray

    bounds: NDArray

    scaling: NDArray
    offset: NDArray

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
    def bounds(self):
        array_bounds = np.vstack((np.zeros(self.ndim), self.shape))

        return self.grid_to_shift(array_bounds)

    @property
    def inverse_scaling(self):
        return 1 / self.scaling

    @property
    def inverse_offset(self):
        return -self.offset / self.scaling

    def grid_to_shift(self, x):
        return x * self.scaling + self.offset

    def shift_to_grid(self, x):
        return x * self.inverse_scaling + self.inverse_offset

    @classmethod
    def from_reference(
        cls, shape, spectral_width, spectrometer_frequency, ref_coord, ref_shift
    ):
        w = np.array(spectral_width)
        N = np.array(shape)
        f = np.array(spectrometer_frequency)

        delta_0 = np.array(ref_shift)
        i_0 = np.array(ref_coord)

        scaling = w / (N * f)
        offset = delta_0 - scaling * i_0

        return cls(ndim=len(shape), shape=shape, scaling=scaling, offset=offset)
