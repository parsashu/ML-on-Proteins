import pandas as pd
import numpy as np

dataset = "datasets/protein_dataset.tsv"
dataset_null_handled = "datasets/protein_dataset_null_handled.tsv"

df = pd.read_csv(dataset, sep="\t")

df.replace("-", np.nan, inplace=True)

df["ASA"] = df["ASA"].fillna(-1)  # Flag
df["pH"] = df["pH"].fillna(7.0)  # Mode
df["T_(C)"] = df["T_(C)"].fillna(25.0)  # Mode
df["REVERSIBILITY"] = df["REVERSIBILITY"].fillna(-1)  # Flag

df["ASA"] = df["ASA"].astype(float)
df["pH"] = df["pH"].astype(float)
df["T_(C)"] = df["T_(C)"].astype(float)
df["REVERSIBILITY"] = df["REVERSIBILITY"].astype(float)

df.to_csv(dataset_null_handled, sep="\t", index=False)
print(
    f"Null values have been handled successfully and saved to: {dataset_null_handled}"
)
