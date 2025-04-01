import pandas as pd
import numpy as np
import re

df = pd.read_csv("datasets/protein_dataset.tsv", sep="\t")
embeddings_df = pd.read_csv("datasets/sequence_embeddings.csv")

embeddings_df["Embedding"] = embeddings_df["Embedding"].apply(
    lambda x: np.array([float(num) for num in re.findall(r"-?\d+\.?\d*e?[-+]?\d*", x)])
)

features = []

for _, row in df.iterrows():
    asa = row["ASA"]
    ph = row["pH"]
    temp = row["T_(C)"]

    sec_str_encoded = eval(row["SEC_STR_ENCODED"])

    protein_seq = row["Protein_Sequence"]
    embedding_row = embeddings_df[embeddings_df["Protein_Sequence"] == protein_seq]

    if not embedding_row.empty:
        embedding = embedding_row["Embedding"].values[0]
    else:
        print(f"No embedding found for protein sequence: {protein_seq}")
        embedding = np.zeros(320)

    feature_vector = [asa, ph, temp] + sec_str_encoded + embedding.tolist()
    features.append(feature_vector)

basic_columns = ["ASA", "pH", "T", "Coil", "Helix", "Sheet", "Turn"]
embedding_columns = [f"embed_{i}" for i in range(320)]
all_columns = basic_columns + embedding_columns

features_df = pd.DataFrame(
    features,
    columns=all_columns,
)

features_df.to_csv("features.csv", index=False)

print("Features shape:", features_df.shape)
print("\nFirst few rows (showing only basic features):")
print(features_df[basic_columns].head())
