import pandas as pd
import numpy as np


# One-hot encode the secondary structure types
df = pd.read_csv("datasets/sequences_filtered_1024.tsv", sep="\t")

sec_str_types = df["SEC_STR"].unique()
sec_str_mapping = {ss_type: i for i, ss_type in enumerate(sec_str_types)}
sec_str_indices = df["SEC_STR"].map(sec_str_mapping)

sec_str_encoded = np.zeros((len(df), len(sec_str_types)), dtype=int)
for i, idx in enumerate(sec_str_indices):
    sec_str_encoded[i, idx] = 1

df["SEC_STR_ENCODED"] = [np.array(encoding) for encoding in sec_str_encoded]

df.to_csv("datasets/embedded_1024_encoded.tsv", sep="\t", index=False)


# Decode the one-hot encoded secondary structure types
def decode_sec_str(encoding):
    if isinstance(encoding, list):
        encoding = np.array(encoding)

    if 1 in encoding:
        idx = np.argmax(encoding)
        return sec_str_types[idx]
    else:
        return None


print("\nSecondary Structure Types and Their Encodings:")
print("-" * 50)
print(f"{'Secondary Structure':<25} {'One-Hot Encoding'}")
print("-" * 50)
for ss_type, idx in sec_str_mapping.items():
    encoding = np.zeros(len(sec_str_types), dtype=int)
    encoding[idx] = 1
    print(f"{ss_type:<25} {encoding}")

print("\nDecoding Example:")
print("-" * 50)

encoding = np.array([0, 1, 0, 0])
decoded = decode_sec_str(encoding)
print(f"Encoding {encoding} decodes to: {decoded}")
