import pandas as pd
import numpy as np
import re
import sys
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


seq_extracted = "../datasets/raw/merged_dataset.tsv"
dataset = "../datasets/protein_dataset.tsv"
all_features = "../datasets/features.csv"
train_path = "../datasets/train_dataset.tsv"
test_path = "../datasets/test_dataset.tsv"
train_augmented = "../datasets/train_dataset_augmented.tsv"
embeddings_train = "../datasets/embeddings_train.csv"
embeddings_test = "../datasets/embeddings_test.csv"

train_processed = "../datasets/train_processed.csv"
test_processed = "../datasets/test_processed.csv"


def merge_embeddings(dataset_path, embeddings_path, output_path):
    df = pd.read_csv(dataset_path, sep="\t", low_memory=False)
    embeddings_df = pd.read_csv(embeddings_path)

    embeddings_df["Embedding"] = embeddings_df["Embedding"].apply(
        lambda x: np.array(
            [float(num) for num in re.findall(r"-?\d+\.?\d*e?[-+]?\d*", x)]
        )
    )

    features = []
    print(f"Processing {len(df)} protein records...")

    for i, (_, row) in enumerate(
        tqdm(df.iterrows(), total=len(df), desc="Processing records")
    ):
        asa = row["ASA"]
        ph = row["pH"]
        temp = row["T_(C)"]

        tm = row["Tm_(C)"] if not pd.isna(row["Tm_(C)"]) else None
        m_value = row["m_(kcal/mol/M)"] if not pd.isna(row["m_(kcal/mol/M)"]) else None
        cm = row["Cm_(M)"] if not pd.isna(row["Cm_(M)"]) else None
        delta_g_h2o = (
            row["∆G_H2O_(kcal/mol)"] if not pd.isna(row["∆G_H2O_(kcal/mol)"]) else None
        )

        sec_str_encoded = eval(row["SEC_STR_ENCODED"])

        protein_seq = row["Protein_Sequence"]
        embedding_row = embeddings_df[embeddings_df["Protein_Sequence"] == protein_seq]

        if not embedding_row.empty:
            embedding = embedding_row["Embedding"].values[0]
        else:
            print(f"No embedding found for protein sequence: {protein_seq}")
            embedding = np.zeros(320)

        feature_vector = (
            [asa, ph, temp]
            + sec_str_encoded
            + [tm, m_value, cm, delta_g_h2o]
            + embedding.tolist()
        )
        features.append(feature_vector)

    basic_columns = ["ASA", "pH", "T_(C)", "Coil", "Helix", "Sheet", "Turn"]
    additional_columns = ["Tm_(C)", "m_(kcal/mol/M)", "Cm_(M)", "∆G_H2O_(kcal/mol)"]
    embedding_columns = [f"embed_{i}" for i in range(320)]
    all_columns = basic_columns + additional_columns + embedding_columns

    features_df = pd.DataFrame(
        features,
        columns=all_columns,
    )

    features_df.to_csv(output_path, index=False)
    print("Features shape:", features_df.shape)


# merge_embeddings(
#     dataset_path=train_augmented,
#     embeddings_path=embeddings_train,
#     output_path=train_processed,
# )
merge_embeddings(
    dataset_path=test_path, embeddings_path=embeddings_test, output_path=test_processed
)
