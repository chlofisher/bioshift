from abc import ABC, abstractmethod
from typing import Callable
from numpy.typing import NDArray


class SpectrumDataSource(ABC):
    cache: NDArray = None

    def get_data(self) -> NDArray:
        if self.cache is None:
            self.cache = self._load_data()

        return self.cache

    @abstractmethod
    def _load_data(self) -> NDArray: ...


class TransformedDataSource(SpectrumDataSource):
    parent: SpectrumDataSource
    func: Callable

    def __init__(self, parent, func):
        self.parent = parent
        self.func = func

    def _load_data(self) -> NDArray:
        return self.func(self.parent.get_data())


class SumDataSource(SpectrumDataSource):
    def __init__(self, source1, source2):
        self.source1 = source1
        self.source2 = source2

    def _load_data(self):
        return self.source1.get_data() + self.source2.get_data()
