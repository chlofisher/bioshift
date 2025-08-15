from numpy.typing import NDArray

from bioshift.core.spectrum import Spectrum
from bioshift.core.peaklist import PeakList

class PeakPicker:
    def pick(self, spectrum: Spectrum) -> NDArray: ...



