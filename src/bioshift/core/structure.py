from Bio.Seq import Seq
from Bio import PDB
from pathlib import Path
import numpy as np


# fmt: off
RESIDUE_ATOMS = {
    "A": ["N", "CA", "C", "O", "CB", "OXT", "H", "H2", "HA", "HB1", "HB2", "HB3", "HXT"],
    "R": ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2", "OXT", "H", "H2", "HA", "HB1", "HB2", "HG1", "HG2", "HD1", "HD2", "HE", "HE1", "HE2", "HZ1", "HZ2", "HZ3", "HXT"],
    "N": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2", "OXT", "H", "H2", "HA", "HB1", "HB2", "HD21", "HD22", "HXT"],
    "D": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2", "OXT", "H", "H2", "HA", "HB1", "HB2", "HXT"],
    "C": ["N", "CA", "C", "O", "CB", "SG", "OXT", "H", "H2", "HA", "HB1", "HB2", "HG", "HXT"],
    "Q": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2", "OXT", "H", "H2", "HA", "HB1", "HB2", "HG1", "HG2", "HE21", "HE22", "HXT"],
    "E": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2", "OXT", "H", "H2", "HA", "HB1", "HB2", "HG1", "HG2", "HXT"],
    "G": ["N", "CA", "C", "O", "OXT", "H", "H2", "HA1", "HA2", "HXT"],
    "H": ["N", "CA", "C", "O", "CB", "CG", "ND1", "CE1", "NE2", "OXT", "H", "H2", "HA", "HB1", "HB2", "HD1", "HE1", "HE2", "HXT"],
    "I": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1", "OXT", "H", "H2", "HA", "HB", "HG11", "HG12", "HG13", "HD11", "HD12", "HD13", "HXT"],
    "L": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "OXT", "H", "H2", "HA", "HB", "HG", "HD11", "HD12", "HD13", "HXT"],
    "K": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ", "OXT", "H", "H2", "HA", "HB1", "HB2", "HG1", "HG2", "HD1", "HD2", "HE1", "HE2", "HZ1", "HZ2", "HZ3", "HXT"],
    "M": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE", "OXT", "H", "H2", "HA", "HB1", "HB2", "HG1", "HG2", "HE1", "HE2", "HXT"],
    "F": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OXT", "H", "H2", "HA", "HB1", "HB2", "HD1", "HD2", "HE1", "HE2", "HZ", "HXT"],
    "P": ["N", "CA", "C", "O", "CB", "CG", "CD", "OXT", "H", "H2", "HA", "HB1", "HB2", "HG1", "HG2", "HD1", "HD2", "HXT"],
    "S": ["N", "CA", "C", "O", "CB", "OG", "OXT", "H", "H2", "HA", "HB1", "HB2", "HG", "HXT"],
    "T": ["N", "CA", "C", "O", "CB", "OG1", "CG2", "OXT", "H", "H2", "HA", "HB", "HG1", "HG21", "HG22", "HG23", "HXT"],
    "W": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2", "OXT", "H", "H2", "HA", "HB1", "HB2", "HD1", "HE1", "HE2", "HE3", "HZ2", "HZ3", "HH2", "HXT"],
    "Y": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH", "OXT", "H", "H2", "HA", "HB1", "HB2", "HD1", "HE1", "HE2", "HE3", "HZ", "HXT"],
    "V": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "OXT", "H", "H2", "HA", "HB", "HG11", "HG12", "HG13", "HXT"],
}
# fmt: on


def init_protein_structure(sequence: Seq, name: str = "protein_structure"):
    structure = PDB.Structure.Structure(name)

    model = PDB.Model.Model(0)
    structure.add(model)

    chain = PDB.Chain.Chain("A")
    model.add(chain)

    segid = ""
    atom_serial = 1

    for i, residue_name in enumerate(sequence):
        residue_id = (" ", i + 1, " ")
        residue = PDB.Residue.Residue(residue_id, residue_name, segid)
        chain.add(residue)

        for atom_name in RESIDUE_ATOMS.get(residue_name, []):
            empty_coord = np.array([np.nan, np.nan, np.nan])

            atom = PDB.Atom.Atom(
                name=atom_name,
                coord=empty_coord,
                bfactor=1.0,
                occupancy=1.0,
                altloc="",
                fullname=f" {atom_name} ",
                serial_number=atom_serial,
                element=atom_name[0],
            )
            atom_serial += 1
            residue.add(atom)

    return structure


def import_protein_structure(path: Path):
    parser = PDB.PDBParser()
    return parser.get_structure(path)
