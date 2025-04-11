import pandas as pd

protein_seq_df = pd.read_csv(
    "datasets/raw/protein_seq.tsv",
    sep="\t",
    dtype={
        "UniProt_ID": str,
        "PDB_wild": str,
        "SEC_STR": str,
        "ASA": str,
        "pH": float,
        "T_(C)": str,
        "Tm_(C)": str,
        "∆Tm_(C)": str,
        "∆H_(kcal/mol)": str,
        "∆Cp_(kcal/mol)": str,
        "∆HvH_(kcal/mol)": str,
        "∆G_(kcal/mol)": str,
        "∆∆G_(kcal/mol)": str,
        "m_(kcal/mol/M)": str,
        "Cm_(M)": str,
        "∆G_H2O_(kcal/mol)": str,
        "∆∆G_H2O_(kcal/mol)": str,
        "STATE": str,
        "REVERSIBILITY": str,
        "Protein_Sequence": str,
    },
    low_memory=False,
)
fireprot_df = pd.read_csv(
    "datasets/raw/fireprotdb_results.csv",
    dtype={
        "ddG": float,
        "dTm": float,
        "tm": float,
        "asa": float,
        "b_factor": float,
        "position": int,
        "conservation": float,
        "is_essential": bool,
        "is_back_to_consensus": bool,
        "is_in_catalytic_pocket": bool,
        "is_in_tunnel_bottleneck": bool,
        "method": str,
        "method_details": str,
        "technique": str,
        "technique_details": str,
        "notes": str,
    },
    low_memory=False,
)
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

kaggle_df["Tm_(C)"] = kaggle_df["tm"]
kaggle_df["Protein_Sequence"] = kaggle_df["protein_sequence"]
kaggle_df = kaggle_df[
    [
        "pH",
        "Tm_(C)",
        "Protein_Sequence",
    ]
]

merged_df = pd.concat([protein_seq_df, fireprot_df, kaggle_df], ignore_index=True)
merged_df.to_csv("datasets/merged_dataset.tsv", sep="\t", index=False)
