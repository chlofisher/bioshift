from abc import ABC, abstractmethod
from typing import Callable
from numpy.typing import NDArray
import math


class SpectrumDataSource(ABC):
    dtype: str
    _cache: NDArray = None

    def get_data(self) -> NDArray:
        if self._cache is None:
            self._cache = self._load_data()

        return self._cache

    @abstractmethod
    def _load_data(self) -> NDArray: ...


class SliceDataSource(SpectrumDataSource):
    parent: SpectrumDataSource
    axis: int
    level: float

    def __init__(self, parent, axis, level):
        self.parent = parent
        self.axis = axis
        self.level = level

    def _load_data(self) -> NDArray:
        parent_data = self.parent.get_data()

        floor = (math.floor(self.level),)
        ceil = (math.ceil(self.level),)
        frac = self.level - floor

        below = parent_data.take(floor, axis=self.axis).squeeze(self.axis)
        above = parent_data.take(ceil, axis=self.axis).squeeze(self.axis)

        return below * (1 - frac) + above * frac


class TransformedDataSource(SpectrumDataSource):
    parent: SpectrumDataSource
    func: Callable

    def __init__(self, parent, func):
        self.parent = parent
        self.func = func
        self.dtype = parent.dtype

    def _load_data(self) -> NDArray:
        return self.func(self.parent.get_data())


class SumDataSource(SpectrumDataSource):
    def __init__(self, source1, source2):
        self.source1 = source1
        self.source2 = source2

        if source1.dtype != source2.dtype:
            raise ValueError("Mismatched dtypes")

        self.dtype = source1.dtype

    def _load_data(self):
        return self.source1.get_data() + self.source2.get_data()
