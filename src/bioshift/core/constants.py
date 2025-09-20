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
