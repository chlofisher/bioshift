"""
Chemical shift data is collected from the Biological Magnetic Resonance Data Bank (BMRB).

Hoch JC, Baskaran K, Burr H, Chin J, Eghbalnia HR, Fujiwara T, Gryk MR, Iwata T, Kojima C, Kurisu G *et al* (2023) Biological Magnetic Resonance Data Bank. *Nucleic Acids Research* 51: D368-76. doi: [10.1093/nar/gkac1050](https://doi.org/10.1093/nar/gkac1050)
"""

import requests
import numpy as np
from rich.progress import Progress
import sys

AMINO_ACIDS_1 = [
    "A",  # Alanine
    "R",  # Arginine
    "N",  # Asparagine
    "D",  # Aspartic acid
    "C",  # Cysteine
    "E",  # Glutamic acid
    "Q",  # Glutamine
    "G",  # Glycine
    "H",  # Histidine
    "I",  # Isoleucine
    "L",  # Leucine
    "K",  # Lysine
    "M",  # Methionine
    "F",  # Phenylalanine
    "P",  # Proline
    "S",  # Serine
    "T",  # Threonine
    "W",  # Tryptophan
    "Y",  # Tyrosine
    "V",  # Valine
]

AMINO_ACIDS_3 = [
    "Ala",
    "Arg",
    "Asn",
    "Asp",
    "Cys",
    "Gln",
    "Glu",
    "Gly",
    "His",
    "Ile",
    "Leu",
    "Lys",
    "Met",
    "Phe",
    "Pro",
    "Ser",
    "Thr",
    "Trp",
    "Tyr",
    "Val",
]

ATOMS = ["H", "N", "CA", "CB"]
ATOMS_GLY = ["H", "N", "CA"]
BMRB_API_URL = "http://api.bmrb.io/current/search/chemical_shifts"
MAX_Z_SCORE = 6

def main() -> None:
    save_path = sys.argv[1]
    save_shift_tables(save_path)


def download_shift_data(atoms: list[str]):
    raw_json = {}
    with Progress() as progress:
        task = progress.add_task("Downloading...", total=len(AMINO_ACIDS_3))

        for aa in AMINO_ACIDS_3:
            # construct the API query URL to fetch the
            # H, N, CA, and CB atoms for each amino acid.
            url = f"{BMRB_API_URL}?comp_id={aa}{"&atom_id=".join(['', *atoms])}"
            print(f"Requesting data from {url}")

            raw_json[aa] = requests.get(url).json()

            progress.update(task, advance=1)

    return raw_json


def save_shift_tables(path: str):
    shift_data = download_shift_data(ATOMS)
    shift_tables = {}

    for aa in AMINO_ACIDS_3:
        if aa == "Pro":
            continue

        atoms = ATOMS if aa != "Gly" else ATOMS_GLY
        table = get_shift_table(raw_json=shift_data, aa=aa, atoms=atoms, z_max=MAX_Z_SCORE)
        shift_tables[aa] = table

    np.savez(path, **shift_tables)
    return shift_tables


def get_shift_table(raw_json: dict, aa: str, atoms: list[str], z_max: float = np.inf):
    data = raw_json[aa]["data"]

    # Every entry in the .json file is assigned a key comprised of its
    # Entry_ID and Comp_index_ID.
    # This is unique for each residue in each protein.
    entry_data = {}
    for entry in data:
        entry_id = entry[0]
        residue_index = entry[3]
        atom_name = entry[5]
        value = entry[7]

        key = (entry_id, residue_index)

        if key in entry_data.keys():
            entry_data[key][atom_name] = value
        else:
            entry_data[key] = {atom_name: value}

    # remove entries which do not have all of the keys for each atom
    entry_data = {
        key: entry
        for key, entry in entry_data.items()
        if set(entry.keys()) == set(atoms)
    }

    # unpack the dictionary into a list of lists and convert to an np array.
    shift_table = [[entry[atom] for atom in atoms]
        for key, entry in entry_data.items()]

    shift_table = np.array(shift_table).T

    # remove columns containing outlier shifts,
    # which are defined as being more than z_max standard deviations from the mean.
    filtered_shifts = []
    for shifts in shift_table:
        std = np.std(shifts)
        mean = np.mean(shifts)

        shifts = np.where(np.abs(shifts - mean) <= z_max * std, shifts, np.nan)
        filtered_shifts.append(shifts)

    shift_table = np.vstack(filtered_shifts)
    shift_table = shift_table[:, ~np.isnan(shift_table).any(axis=0)]

    return shift_table


if __name__ == "__main__":
    main()
