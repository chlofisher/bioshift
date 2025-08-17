from enum import Enum


class NMRNucleus(Enum):
    """
    Enum representing the NMR-active nucleus associated with a particular chemical shift value.
    """
    HYDROGEN = "1H"
    NITROGEN = "15N"
    CARBON = "13C"
