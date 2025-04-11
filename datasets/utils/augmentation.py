import random
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
}

blosum62 = substitution_matrices.load("BLOSUM62")
amino_acids = list(hydrophobicity.keys())


def substitute_with_constraints(
    sequence, num_substitutions=3, min_score=1.0, max_hydro_diff=1.0
):
    """Substitute amino acids using BLOSUM62 and hydrophobicity constraints."""
    seq_list = list(sequence)
    positions = random.sample(range(len(seq_list)), num_substitutions)

    for pos in positions:
        old_aa = seq_list[pos]
        old_hydro = hydrophobicity[old_aa]

        candidates = [
            aa
            for aa in amino_acids
            if blosum62[old_aa][aa] >= min_score
            and abs(hydrophobicity[aa] - old_hydro) <= max_hydro_diff
            and aa != old_aa
        ]
        if candidates:
            new_aa = random.choice(candidates)
            seq_list[pos] = new_aa

    new_seq = "".join(seq_list)
    return new_seq


original_seq = "MAKVRTKDVMEQFNLELISGEEGINRPITMSDLSRPGIEIAGYFTYYPRERVQLLGK"
new_seq = substitute_with_constraints(
    original_seq, num_substitutions=3, min_score=1.0, max_hydro_diff=1.0
)
print(f"Original: {original_seq}")
print(f"Modified: {new_seq}")
print("equal: ", original_seq == new_seq)


# amino_acids_ordered = sorted(blosum62.alphabet)
# plt.figure(figsize=(12, 10))
# matrix_data = np.array(
#     [[blosum62[aa1][aa2] for aa2 in amino_acids_ordered] for aa1 in amino_acids_ordered]
# )
# sns.heatmap(
#     matrix_data,
#     annot=True,
#     fmt=".2f",
#     cmap="viridis",
#     xticklabels=amino_acids_ordered,
#     yticklabels=amino_acids_ordered,
# )
# plt.title("BLOSUM62 Substitution Matrix")
# plt.tight_layout()
# plt.show()
