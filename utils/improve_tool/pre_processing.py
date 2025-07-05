import pandas as pd
import sys
import os
sys.path.append(os.path.abspath("."))
from utils.embedding.embedding_functions import embed_dataset_without_save
from utils.embedding.merge_embeddings import merge_embeddings_without_save
from utils.improve_tool.mutation_funs import approved_mutated_sequences


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
    for seq in mutated_seqs:
        row = base_features.copy()
        row["Protein_Sequence"] = seq
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def pre_processing(df):
    embeddings_df = embed_dataset_without_save(df)
    X = merge_embeddings_without_save(df, embeddings_df)
    return X


json_data = {
    "ASA": 9.0,
    "pH": 6.0,
    "Tm_(C)": 80.4,
    "Coil": 1,
    "Helix": 0,
    "Sheet": 0,
    "Turn": 0,
    "Protein_Sequence": "MGDVEKGKKIFVQKCAQCHTVEKGGKHKTGPNLHGLFGRKTGQAPGFTYTDANKNKGITWKEETLMEYLENPKKYIPGTKMIFAGIKKKTEREDLIAYLKKATNE",
}

df = original_to_mutated(json_data)
print(df)

