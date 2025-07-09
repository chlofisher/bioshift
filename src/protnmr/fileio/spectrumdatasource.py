from abc import ABC, abstractmethod
from numpy.typing import NDArray


class SpectrumDataSource(ABC):
    """Class which lazily reads raw spectrum data.

    Attributes:
        cache: Stores a cache of the raw spectrum array after it is loaded.
    """
    cache: NDArray

    def get_data(self) -> NDArray:
        """Checks if the data is in the cache, and loads it if not.

        Returns:
            Raw spectrum data array.
        """
        if self.cache is None:
            self.cache = self._load_data()

        return self.cache

    @abstractmethod
    def _load_data(self) -> NDArray:
        """Load the raw data into an NDArray.

        Returns:
            Raw spectrum data array.
        """
        ...
