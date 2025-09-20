from importlib import resources
import numpy as np
from numpy.typing import NDArray
from scipy import stats

from bioshift.core.constants import AMINO_ACIDS_3


def predict_amino_acid(spin_system: dict[str, float]) -> NDArray:
    # Keys for rows in the shift_tables
    # TODO: Store this in the .npz file
    ATOM_KEYS = ["H", "N", "CA", "CB"]

    row_indices = [ATOM_KEYS.index(key) for key in spin_system.keys()]

    shift_array = np.array(list(spin_system.values()))
    print(shift_array)

    # Check if there is a beta carbon shift, as this precludes glycine.
    cb_shift: bool = "CB" in spin_system.keys()

    n = 20

    p_delta = 0
    p_delta_given_aa = np.zeros(n)

    with resources.open_binary("bioshift.data", "shift_tables.npz") as f:
        shift_tables = np.load(f)

        for i, aa in enumerate(AMINO_ACIDS_3):
            is_proline: bool = aa == "Pro"
            is_glycine: bool = aa == "Gly"

            if is_proline or (is_glycine and cb_shift):
                continue

            table: NDArray = shift_tables[aa][row_indices]

            kde = stats.gaussian_kde(table, bw_method="silverman")
            prob_density = kde(shift_array).item()

            p_delta += prob_density
            p_delta_given_aa[i] = prob_density

    if np.close(p_delta, 0):
        return None, p_delta

    p_aa_given_delta = p_delta_given_aa / p_delta

    return p_aa_given_delta, p_delta
