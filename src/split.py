import pandas as pd


dataset_path = "datasets/protein_dataset.tsv"
train_path = "datasets/train_dataset.tsv"
test_path = "datasets/test_dataset.tsv"

df = pd.read_csv(dataset_path, sep="\t", low_memory=False)

test_df = pd.DataFrame(columns=df.columns).astype(df.dtypes)
remaining_df = df.copy()

# Target test size (14% of total dataset)
target_test_size = int(0.14 * len(df))

while len(test_df) < target_test_size and len(remaining_df) > 0:
    if len(test_df) % 3000 == 0 and len(test_df) != 0:
        print(
            f"Progress: {len(test_df)}/{target_test_size} test samples selected ({len(test_df)/target_test_size*100:.1f}%)"
        )
    # Get group with same sequence
    random_seq = remaining_df["Protein_Sequence"].sample(1, random_state=42).iloc[0]
    sequence_samples = remaining_df[remaining_df["Protein_Sequence"] == random_seq]

    # Check if all samples have Tm, pH values
    if (
        not sequence_samples["Tm_(C)"].isna().any()
        and not sequence_samples["pH"].isna().any()
    ):
        test_df = pd.concat([test_df, sequence_samples], ignore_index=True)
        # Remove these samples from remaining dataset preventing data leakage
        remaining_df = remaining_df[remaining_df["Protein_Sequence"] != random_seq]

train_df = remaining_df

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
