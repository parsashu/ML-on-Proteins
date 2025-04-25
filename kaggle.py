import pandas as pd
import numpy as np

seq_extracted = "datasets/raw/merged_dataset.tsv"
dataset = "datasets/protein_dataset.tsv"
all_features = "datasets/features.csv"
train_path = "datasets/train_dataset.tsv"
test_path = "datasets/test_dataset.tsv"


df = pd.read_csv(dataset, sep="\t", low_memory=False)

# Extract Kaggle test sequences
kaggle_sequence = "VPVNPEPDATSVENVAEKTGSGDSQSDPIKADLEVKGQSALPFDVDCWAILCKGAPNVLQRVNEKTKNSNRDRSGANKGPFKDPQKWGIKALPPKNPSWSAQDFKSPEEYAFASSLQGGTNAILAPVNLASQNSQGGVLNGFYSANKVAQFDPSKPQQTKGTWFQITKFTGAAGPYCKALGSNDKSVCDKNKNIAGDWGFDPAKWAYQYDEKNNKFNYVGK"
kaggle_index = df[df["Protein_Sequence"] == kaggle_sequence].index[0]
kaggle_test_df = df.iloc[kaggle_index:]

reduced_df = df.iloc[:kaggle_index]
kaggle_unique_sequences = kaggle_test_df["Protein_Sequence"].nunique()
total_unique_sequences = df["Protein_Sequence"].nunique()
kaggle_percentage = (kaggle_unique_sequences / total_unique_sequences) * 100
print(
    f"Kaggle test unique sequences: {kaggle_unique_sequences} ({kaggle_percentage:.1f}%)"
)

# Target test size: 20% of unique sequences
target_test_size = int(
    (0.21 - kaggle_percentage / 100) * len(reduced_df["Protein_Sequence"].unique())
)

# Group by protein sequence and filter out groups with missing values
valid_groups = reduced_df.groupby("Protein_Sequence").filter(
    lambda x: not x["Tm_(C)"].isna().any() and not x["pH"].isna().any()
)

unique_sequences = valid_groups["Protein_Sequence"].unique()
np.random.seed(0)
selected_sequences = np.random.choice(
    unique_sequences, size=target_test_size, replace=False
)

# Create Train dataset without selected groups to prevent data leakage
test_df = valid_groups[valid_groups["Protein_Sequence"].isin(selected_sequences)]
train_df = reduced_df[~reduced_df["Protein_Sequence"].isin(selected_sequences)]
test_df = pd.concat([test_df, kaggle_test_df], ignore_index=True)

train_df.to_csv(train_path, sep="\t", index=False)
test_df.to_csv(test_path, sep="\t", index=False)

print(f"Total samples: {len(df)}")
print(f"Train samples: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
print(f"Test samples: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")

print(
    f"Unique sequences in train set: {train_df['Protein_Sequence'].nunique()} ({train_df['Protein_Sequence'].nunique()/df['Protein_Sequence'].nunique()*100:.1f}%)"
)
print(
    f"Unique sequences in test set: {test_df['Protein_Sequence'].nunique()} ({test_df['Protein_Sequence'].nunique()/df['Protein_Sequence'].nunique()*100:.1f}%)"
)
