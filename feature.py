import pandas as pd


def remove_features(file_path):
    df = pd.read_csv(file_path, sep="\t")

    selected_columns = [
        "ASA",
        "pH",
        "Tm_(C)",
        "Protein_Sequence",
        "SEC_STR",
        "SEC_STR_ENCODED",
    ]
    df_selected = df[selected_columns]
    df_selected.to_csv(file_path, sep="\t", index=False)
