# Single-letter amino acid codes
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

# Three-letter amino acid codes
AMINO_ACIDS_3 = [
    "Ala",  # Alanine
    "Arg",  # Arginine
    "Asn",  # Asparagine
    "Asp",  # Aspartic acid
    "Cys",  # Cysteine
    "Glu",  # Glutamic acid
    "Gln",  # Glutamine
    "Gly",  # Glycine
    "His",  # Histidine
    "Ile",  # Isoleucine
    "Leu",  # Leucine
    "Lys",  # Lysine
    "Met",  # Methionine
    "Phe",  # Phenylalanine
    "Pro",  # Proline
    "Ser",  # Serine
    "Thr",  # Threonine
    "Trp",  # Tryptophan
    "Tyr",  # Tyrosine
    "Val",  # Valine
]

AMINO_ACIDS_3_TO_1 = dict(zip(AMINO_ACIDS_3, AMINO_ACIDS_1))
AMINO_ACIDS_1_TO_3 = dict(zip(AMINO_ACIDS_1, AMINO_ACIDS_3))

"""
Statistics collected from UniProtKB
https://www.uniprot.org/uniprotkb/statistics
Non-standard amino acids were removed, and percentages renormalized.
"""
AMINO_ACID_FREQUENCY = {
    "Leu": 0.09813,
    "Ala": 0.08913,
    "Gly": 0.07222,
    "Ser": 0.06942,
    "Val": 0.06842,
    "Glu": 0.06272,
    "Arg": 0.05852,
    "Thr": 0.05562,
    "Ile": 0.05482,
    "Asp": 0.05482,
    "Pro": 0.05052,
    "Lys": 0.04971,
    "Phe": 0.03881,
    "Gln": 0.03821,
    "Asn": 0.03821,
    "Tyr": 0.02871,
    "Met": 0.02331,
    "His": 0.02241,
    "Cys": 0.01330,
    "Trp": 0.01300,
}
