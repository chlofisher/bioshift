from __future__ import annotations
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
    scaling: NDArray
    offset: NDArray

    inverse: SpectrumTransform

    def __init__(self, scaling, offset):
        """Initialise the SpectrumTransform. Performs validation logic and 
        copies the input arrays to ensure immutability.

        Raises:
            ValueError: If any of the scaling values are zero,
                to enforce that the transformation matrix is non-singular.
            ValueError: If the shape of the scaling and the offset vectors do 
                not match.
        """
        if not all(x != 0 for x in scaling):
            raise ValueError("""Non-invertible transformation: 
                SpectrumTransform scaling values must all be non-zero.
                """ .strip())

        if scaling.shape != offset.shape:
            raise ValueError(
                "Scaling and offset vectors must have the same shape")

        self.scaling = np.array(scaling, dtype=float).copy()
        self.offset = np.array(offset, dtype=float).copy()

    def __eq__(self, other):
        if not isinstance(other, SpectrumTransform):
            return NotImplemented

        return (np.allclose(self.scaling, other.scaling)
                and np.allclose(self.offset, other.offset))

    def __hash__(self):
        return hash((self.scaling.tobytes(), self.offset.tobytes()))

    @property
    def inverse(self):
        inverse_scaling = 1/self.scaling

        inverse_offset = -(inverse_scaling * self.offset)

        return SpectrumTransform(inverse_scaling, inverse_offset)

    @property
    def shape(self):
        return self.offset.shape

    def apply(self, point: NDArray) -> NDArray:
        """Apply the transformation to a point.

        Returns:
            Transformed point as a vector.

        Raises:
            ValueError: If the shapes of the input point and the transform do 
                not match.
        """
        if self.shape != point.shape:
            raise ValueError(f"""Shape of point {point.shape} must match shape
                of transform {self.shape}""")
        return point * self.scaling + self.offset
