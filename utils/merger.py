import pandas as pd

protein_seq_df = pd.read_csv("datasets/raw/protein_seq.tsv", sep="\t")
fireprot_df = pd.read_csv("datasets/raw/fireprotdb_results.csv", dtype=str)
kaggle_df = pd.read_csv("datasets/raw/kaggle_data.csv")


def translate_sec_str(code):
    if code == "H" or code == "G" or code == "I":
        return "Helix"
    elif code == "E" or code == "B":
        return "Sheet"
    elif code == "T" or code == "S":
        return "Turn"
    elif code == "L":
        return "Coil"
    else:
        return "-"


fireprot_df["UniProt_ID"] = fireprot_df["uniprot_id"]
fireprot_df["SEC_STR"] = fireprot_df["secondary_structure"].apply(translate_sec_str)
fireprot_df["ASA"] = fireprot_df["asa"]
fireprot_df["Tm_(C)"] = fireprot_df["tm"]
fireprot_df["∆Tm_(C)"] = fireprot_df["dTm"]
fireprot_df["∆∆G_(kcal/mol)"] = fireprot_df["ddG"]
fireprot_df["Protein_Sequence"] = fireprot_df["sequence"]
fireprot_df = fireprot_df[
    [
        "UniProt_ID",
        "SEC_STR",
        "ASA",
        "pH",
        "Tm_(C)",
        "∆Tm_(C)",
        "∆∆G_(kcal/mol)",
        "Protein_Sequence",
    ]
]

kaggle_df["Tm_(C)"] = kaggle_df["tm"].astype("string")
kaggle_df["Protein_Sequence"] = kaggle_df["protein_sequence"]
kaggle_df = kaggle_df[
    [
        "pH",
        "Tm_(C)",
        "Protein_Sequence",
    ]
]

merged_df = pd.concat([protein_seq_df, fireprot_df, kaggle_df], ignore_index=True)

merged_df = merged_df.fillna("-")
for column in merged_df.columns:
    merged_df[column] = merged_df[column].astype("string")

merged_df.to_csv("datasets/raw/merged_dataset.tsv", sep="\t", index=False)

new_df = pd.read_csv("datasets/raw/merged_dataset.tsv", sep="\t")
non_string_values = new_df.iloc[:, 6][
    ~new_df.iloc[:, 6].apply(lambda x: isinstance(x, str))
]
print(non_string_values)
