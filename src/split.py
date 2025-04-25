import pandas as pd
import numpy as np


dataset = "datasets/protein_dataset.tsv"
train_path = "datasets/train_dataset2.tsv"
test_path = "datasets/test_dataset2.tsv"

# Read the dataset
df = pd.read_csv(dataset, sep="\t", low_memory=False)

# Calculate target test size
target_test_size = int(0.2 * len(df["Protein_Sequence"].unique()))

# Group by protein sequence and filter out groups with missing values
valid_groups = df.groupby("Protein_Sequence").filter(
    lambda x: not x["Tm_(C)"].isna().any() and not x["pH"].isna().any()
)

# Get unique sequences
unique_sequences = valid_groups["Protein_Sequence"].unique()

# Calculate how many sequences we need to reach target test size
np.random.seed(0)
selected_sequences = np.random.choice(
    unique_sequences, size=target_test_size, replace=False
)

# Split the data
test_df = valid_groups[valid_groups["Protein_Sequence"].isin(selected_sequences)]
train_df = df[~df["Protein_Sequence"].isin(selected_sequences)]

# Save the datasets
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
