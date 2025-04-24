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
    "B": -3.5,  # Avg of D (-3.5) and N (-3.5)
    "Z": -3.5,  # Avg of E (-3.5) and Q (-3.5)
    "X": 0.0,   # Unknown, assumed neutral
}


blosum62 = substitution_matrices.load("BLOSUM62")
amino_acids = list(hydrophobicity.keys())


def substitute_with_constraints(
    sequence, num_substitutions=3, min_score=0, max_hydro_diff=1.0
):
    seq_list = list(sequence)
    positions = random.sample(range(len(seq_list)), num_substitutions)

    for pos in positions:
        old_aa = seq_list[pos]
        old_hydro = hydrophobicity[old_aa]
        # Filter by BLOSUM62 score and hydrophobicity
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

    return "".join(seq_list)


def augment_sequence(
    sequence,
    num_substitutions=[1, 2, 3],
    num_mutations=[5, 4, 3],
    min_score=0,
    max_hydro_diff=1.0,
    retries = 3,
    random_seed=None,
):
    """Augment a sequence by substituting amino acids with constraints."""
    random.seed(random_seed)
    new_seq_list = []

    for i in range(len(num_substitutions)):
        current_substitution = num_substitutions[i]
        current_mutation = num_mutations[i]

        for _ in range(current_mutation):
            for j in range(retries):
                new_seq = substitute_with_constraints(
                    sequence,
                    num_substitutions=current_substitution,
                    min_score=min_score,
                    max_hydro_diff=max_hydro_diff,
                )
                if new_seq not in new_seq_list and new_seq != sequence:
                    new_seq_list.append(new_seq)
                    break
            
                if j == retries - 1:
                    print(f"Failed to generate a valid variant after {retries} retries")

    return new_seq_list


original_seq = "MAKVRTKDVMEQFNLELISGEEGINRPITMSDLSRPGIEIAGYFTYYPRERVQLLGK"
new_seqs = augment_sequence(
    original_seq,
    num_substitutions=[1, 2, 3],
    num_mutations=[5, 4, 3],
    min_score=0,
    max_hydro_diff=1,
    random_seed=42,
)

print("\nSequence Augmentation Results:")
print("-" * 50)
print(f"Original Sequence: {original_seq}")
print("-" * 50)
print("Generated Variants:")
for i, seq in enumerate(new_seqs, 1):
    # Find positions where the sequence differs
    diffs = [j for j, (a, b) in enumerate(zip(original_seq, seq)) if a != b]
    print(f"\nVariant {i}:")
    print(f"Sequence: {seq}")
    print(f"Changes: {len(diffs)} substitutions at positions {diffs}")
print("-" * 50)
print(f"Total variants generated: {len(new_seqs)}")
