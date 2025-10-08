from abc import ABC, abstractmethod
from typing import Callable
from numpy.typing import NDArray


class SpectrumDataSource(ABC):
    _cache: NDArray | None = None

    def get_data(self) -> NDArray:
        if self._cache is None:
            self._cache = self._load_data()

        return self._cache

    @abstractmethod
    def _load_data(self) -> NDArray: ...


class TransformedDataSource(SpectrumDataSource):
    parent: SpectrumDataSource
    func: Callable

    def __init__(self, parent, func):
        self.parent = parent
        self.func = func
        self.dtype = parent.dtype

    def _load_data(self) -> NDArray:
        return self.func(self.parent.get_data())
