import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import os


test_path = "datasets/test_dataset.tsv"
train_augmented = "datasets/train_dataset_augmented.tsv"
embeddings_train = "datasets/embeddings_train.csv"
embeddings_val = "datasets/embeddings_val.csv"
embeddings_test = "datasets/embeddings_test.csv"
train_processed = "datasets/train_processed.csv"
val_processed = "datasets/val_processed.csv"
test_processed = "datasets/test_processed.csv"


def merge_embeddings(dataset_path, embeddings_path, output_path, batch_size=50000):
    df = pd.read_csv(dataset_path, sep="\t", low_memory=False)
    embeddings_df = pd.read_csv(embeddings_path)

    embeddings_df["Embedding"] = embeddings_df["Embedding"].apply(
        lambda x: np.array(
            [float(num) for num in re.findall(r"-?\d+\.?\d*e?[-+]?\d*", x)]
        )
    )

    embedding_dict = {}
    for _, row in tqdm(
        embeddings_df.iterrows(),
        total=len(embeddings_df),
        desc="Building embedding dictionary",
    ):
        embedding_dict[row["Protein_Sequence"]] = row["Embedding"]

    features = []
    basic_columns = ["ASA", "pH", "Coil", "Helix", "Sheet", "Turn"]
    additional_columns = ["Tm_(C)"]
    embedding_columns = [f"embed_{i}" for i in range(320)]
    all_columns = basic_columns + additional_columns + embedding_columns

    # Clear the output file if it exists
    if os.path.exists(output_path):
        os.remove(output_path)
    pd.DataFrame(columns=all_columns).to_csv(output_path, index=False)

    for i, (_, row) in enumerate(
        tqdm(df.iterrows(), total=len(df), desc="Processing records")
    ):
        asa = row["ASA"]
        ph = row["pH"]
        tm = row["Tm_(C)"] if not pd.isna(row["Tm_(C)"]) else None

        sec_str_encoded = eval(row["SEC_STR_ENCODED"])
        protein_seq = row["Protein_Sequence"]

        embedding = embedding_dict.get(protein_seq)
        # Not saving samples with no embedding
        if embedding is None:
            print(f"No embedding found for protein sequence: {protein_seq}")
            continue

        feature_vector = [asa, ph] + sec_str_encoded + [tm] + embedding.tolist()
        features.append(feature_vector)

        # Process and save in batches
        if (i + 1) % batch_size == 0 or i == len(df) - 1:
            features_df = pd.DataFrame(features, columns=all_columns)
            features_df.to_csv(
                output_path,
                mode="a",
                header=not os.path.exists(output_path),
                index=False,
            )
            features = []  # Clear the features list


def merge_embeddings_without_save(df, embeddings_df, batch_size=10000):

    embeddings_df["Embedding"] = embeddings_df["Embedding"].apply(
        lambda x: np.array(
            [float(num) for num in re.findall(r"-?\d+\.?\d*e?[-+]?\d*", x)]
        )
    )

    embedding_dict = {}
    for _, row in tqdm(
        embeddings_df.iterrows(),
        total=len(embeddings_df),
        desc="Building embedding dictionary",
    ):
        embedding_dict[row["Protein_Sequence"]] = row["Embedding"]

    features = []
    basic_columns = ["ASA", "pH", "Coil", "Helix", "Sheet", "Turn"]
    embedding_columns = [f"embed_{i}" for i in range(320)]
    all_columns = basic_columns + embedding_columns

    for i, (_, row) in enumerate(
        tqdm(df.iterrows(), total=len(df), desc="Processing records")
    ):
        asa = row["ASA"]
        ph = row["pH"]

        sec_str_encoded = eval(row["SEC_STR_ENCODED"])
        protein_seq = row["Protein_Sequence"]

        embedding = embedding_dict.get(protein_seq)
        # Not saving samples with no embedding
        if embedding is None:
            print(f"No embedding found for protein sequence: {protein_seq}")
            continue

        feature_vector = [asa, ph] + sec_str_encoded + embedding.tolist()
        features.append(feature_vector)

        # Process and save in batches
        if (i + 1) % batch_size == 0 or i == len(df) - 1:
            features_df = pd.DataFrame(features, columns=all_columns)
            features = []  # Clear the features list
            
    return features_df


# merge_embeddings(
#     dataset_path=train_augmented,
#     embeddings_path=embeddings_train,
#     output_path=train_processed,
# )

# merge_embeddings(
#     dataset_path=test_path,
#     embeddings_path=embeddings_test,
#     output_path=test_processed,
# )
