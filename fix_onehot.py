import pandas as pd
import numpy as np

dataset = "datasets/merged_no_duplicates.tsv"

COIL = 0
HELIX = 1
SHEET = 2
TURN = 3


def encode_secondary_structure(structure_text):
    """
    Encode secondary structure information as a count vector.
    Format: [coil_count, helix_count, sheet_count, turn_count]
    """
    if not structure_text or structure_text == "-":
        return [0, 0, 0, 0]

    counts = [0, 0, 0, 0]
    structures = [s.strip() for s in structure_text.split(",")]

    for structure in structures:
        if "Coil" in structure:
            counts[COIL] += 1
        elif "Helix" in structure:
            counts[HELIX] += 1
        elif "Sheet" in structure:
            counts[SHEET] += 1
        elif "Turn" in structure:
            counts[TURN] += 1

    return counts


df = pd.read_csv(dataset, sep="\t")
df["SEC_STR_ENCODED"] = df["SEC_STR"].apply(encode_secondary_structure)
df.to_csv(dataset, sep="\t", index=False)


def decode_sec_str(encoding):
    if not isinstance(encoding, list) and not isinstance(encoding, np.ndarray):
        return None

    structure_types = []
    for i, count in enumerate(encoding):
        if i == COIL and count > 0:
            structure_types.extend(["Coil"] * count)
        elif i == HELIX and count > 0:
            structure_types.extend(["Helix"] * count)
        elif i == SHEET and count > 0:
            structure_types.extend(["Sheet"] * count)
        elif i == TURN and count > 0:
            structure_types.extend(["Turn"] * count)

    return ", ".join(structure_types) if structure_types else None


print("\nSecondary Structure Types and Their Encodings:")
print("-" * 50)
print(f"{'Secondary Structure':<25} {'Count-Based Encoding'}")
print("-" * 50)

examples = [
    "Coil",
    "Helix",
    "Sheet",
    "Turn",
    "Coil, Coil",
    "Helix, Sheet",
    "Helix, Turn, Turn",
    "-",
]

for example in examples:
    encoding = encode_secondary_structure(example)
    print(f"{example:<25} {encoding}")

print("\nDecoding Examples:")
print("-" * 50)

for example in examples:
    encoding = encode_secondary_structure(example)
    decoded = decode_sec_str(encoding)
    print(f"Original: {example:<20} Encoding: {encoding}  Decoded: {decoded}")
