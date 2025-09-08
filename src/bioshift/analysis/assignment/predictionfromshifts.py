from __future__ import annotations
from importlib import resources
import numpy as np
from Bio.Seq import Seq
from Bio.Data.IUPACData import protein_letters_1to3
from scipy.stats import gaussian_kde

from bioshift.core.peak import NMRAtom, NMRResidue


def predict_assignment_from_shifts(residue: NMRResidue, sequence: Seq):
    # Keys for rows in the shift_tables
    # TODO: Store this in the .npz file
    ATOM_KEYS = ["H", "N", "CA", "CB"]

    row_indices = [
        ATOM_KEYS.index(shift.atom.name)
        for shift in residue.shifts
        if shift.atom is not None
    ]

    shift_array = np.array([shift.shift for shift in residue.shifts])

    cb_shift: bool = NMRAtom.CB in [shift.atom for shift in residue.shifts]

    n = len(sequence)

    p_delta = 0
    p_delta_given_res = np.zeros(n)

    with resources.open_binary("bioshift.data", "shift_tables.npz") as f:
        shift_tables = np.load(f)

        for i, aa in enumerate(sequence):
            aa_three_letter = protein_letters_1to3[aa]

            is_proline: bool = aa == "P"
            is_glycine: bool = aa == "G"

            if is_proline or (is_glycine and cb_shift):
                # print(f"Ignoring {aa}{i+1}")
                continue

            table = shift_tables[aa_three_letter][row_indices]

            kde = gaussian_kde(table, bw_method="silverman")
            prob_density = kde(shift_array).item()

            p_delta += prob_density
            p_delta_given_res[i] = prob_density

    p_res_given_delta = p_delta_given_res / p_delta
    return p_res_given_delta
