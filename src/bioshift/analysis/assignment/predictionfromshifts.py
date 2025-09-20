from importlib import resources
import numpy as np
from numpy.typing import NDArray
from scipy import stats
import math

from bioshift.core.constants import AMINO_ACIDS_3


def predict_amino_acid(*spin_systems: dict[str, float]) -> NDArray:
    # Keys for rows in the shift_tables
    # TODO: Store this in the .npz file
    SHIFT_TABLE_ROW_KEYS = ["H", "N", "CA", "CB"]

    atom_keys: list[str] = spin_systems[0].keys()
    same_keys: bool = all(sys.keys() == atom_keys for sys in spin_systems)

    if not same_keys:
        raise ValueError("Spin systems must have matching keys for batch amino acid prediction.")

    row_indices = [SHIFT_TABLE_ROW_KEYS.index(key) for key in atom_keys]

    shift_array = np.array([list(sys.values()) for sys in spin_systems]).T

    # Check if there is a beta carbon shift, as this precludes glycine.
    cb_shift: bool = "CB" in atom_keys

    n = 20

    p_delta = np.zeros(len(spin_systems))
    p_delta_given_aa = np.zeros((len(spin_systems), n))

    with resources.open_binary("bioshift.data", "shift_tables.npz") as f:
        shift_tables = np.load(f)

        for i, aa in enumerate(AMINO_ACIDS_3):
            is_proline: bool = aa == "Pro"
            is_glycine: bool = aa == "Gly"

            if is_proline or (is_glycine and cb_shift):
                continue

            table: NDArray = shift_tables[aa][row_indices]

            kde = stats.gaussian_kde(table, bw_method="silverman")
            prob_density = kde(shift_array)

            p_delta += prob_density
            p_delta_given_aa[:, i] = prob_density

    p_aa_given_delta = p_delta_given_aa / p_delta[:, np.newaxis]

    return p_aa_given_delta, p_delta
