from Bio.Align import substitution_matrices

# Hydrophobicity scale (Kyte-Doolittle)
hydrophobicity = {
    "A": 1.8,
    "C": 2.5,
    "D": -3.5,
    "E": -3.5,
    "F": 2.8,
    "G": 0.4,
    "H": -3.2,
    "I": 4.5,
    "K": -3.9,
    "L": 3.8,
    "M": 1.9,
    "N": -3.5,
    "P": -1.6,
    "Q": -3.5,
    "R": -4.5,
    "S": -0.8,
    "T": -0.7,
    "V": 4.2,
    "W": -0.9,
    "Y": -1.3,
    "B": -3.5,  # Avg of D (-3.5) and N (-3.5)
    "Z": -3.5,  # Avg of E (-3.5) and Q (-3.5)
    "X": 0.0,  # Unknown, assumed neutral
}


blosum62 = substitution_matrices.load("BLOSUM62")
amino_acids = list(hydrophobicity.keys())


def approved_mutated_sequences(sequence, min_score=0, max_hydro_diff=1.0):
    new_seqs_list = []
    seq_list = list(sequence)

    for pos in range(len(seq_list)):
        old_aa = seq_list[pos]
        old_hydro = hydrophobicity[old_aa]
        # Filter by BLOSUM62 score and hydrophobicity
        new_aa_list = [
            aa
            for aa in amino_acids
            if blosum62[old_aa][aa] >= min_score
            and abs(hydrophobicity[aa] - old_hydro) <= max_hydro_diff
            and aa != old_aa
        ]
        if new_aa_list:
            for new_aa in new_aa_list:
                new_seq = seq_list.copy()
                new_seq[pos] = new_aa
                new_seqs_list.append("".join(new_seq))

    return new_seqs_list


# original_seq = "MAKVRTKDVMEQFNLELISGEEGINRPITMSDLSRPGIEIAGYFTYYPRERVQLLGK"
# print(len(original_seq), len(hydrophobicity))
# new_seqs = approved_mutated_sequences(
#     original_seq,
#     min_score=0,
#     max_hydro_diff=1,
# )

# print(new_seqs)
# print(len(new_seqs))
