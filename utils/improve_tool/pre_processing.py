import pandas as pd
import sys
import os

sys.path.append(os.path.abspath("."))
from utils.embedding.embedding_functions import embed_dataset_without_save
from utils.embedding.merge_embeddings import merge_embeddings_without_save
from utils.improve_tool.mutation_funs import approved_mutated_sequences
import joblib


def original_to_mutated(json_data):
    original_seq = json_data["Protein_Sequence"]
    mutated_seqs = approved_mutated_sequences(original_seq)

    base_features = {
        "ASA": json_data["ASA"],
        "pH": json_data["pH"],
        "Coil": json_data["Coil"],
        "Helix": json_data["Helix"],
        "Sheet": json_data["Sheet"],
        "Turn": json_data["Turn"],
    }

    rows = []
    # Add the original sequence first
    original_row = base_features.copy()
    original_row["Protein_Sequence"] = original_seq
    rows.append(original_row)

    # Add mutated sequences
    for seq in mutated_seqs:
        row = base_features.copy()
        row["Protein_Sequence"] = seq
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def normalize(df):
    scaler = joblib.load("./models/scalers/feature_scaler.joblib")
    asa_scaler = joblib.load("./models/scalers/asa_scaler.joblib")
    asa_col = "ASA"
    cols_to_normalize = [
        col
        for col in df.columns
        if col not in ["ASA", "Coil", "Helix", "Sheet", "Turn"]
    ]

    # Normalize ASA separately
    df[asa_col] = asa_scaler.transform(df[[asa_col]])

    # Normalize other features
    df[cols_to_normalize] = scaler.transform(df[cols_to_normalize])
    return df


def pre_processing(json_data):
    df = original_to_mutated(json_data)
    embeddings_df = embed_dataset_without_save(df)
    X = merge_embeddings_without_save(df, embeddings_df)
    X_norm = normalize(X)
    return X_norm
