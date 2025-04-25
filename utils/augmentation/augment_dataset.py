import os
import sys
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from utils.augmentation.augment_functions import augment_sequence


def augment_dataset(
    input_file, output_file, num_substitutions=[1, 2, 3], num_mutations=[5, 4, 3]
):
    """Augment protein sequences in the dataset while preserving other features."""
    df = pd.read_csv(input_file, sep="\t", low_memory=False)

    # Split the dataset into two parts
    df_before = df.iloc[:42948]
    df_after = df.iloc[42948:]

    augmented_data = []
    for i, (_, row) in enumerate(df_before.iterrows()):
        if i % 100 == 0:
            print(f"Processing row {i} of {len(df_before)}")
        augmented_data.append(row)
        original_seq = row["Protein_Sequence"]

        # Generate augmented sequences
        augmented_seqs = augment_sequence(
            original_seq,
            num_substitutions=num_substitutions,
            num_mutations=num_mutations,
            min_score=2,
            max_hydro_diff=0.5,
            retries=20,
            random_seed=42,
        )

        for aug_seq in augmented_seqs:
            new_row = row.copy()
            new_row["Protein_Sequence"] = aug_seq
            augmented_data.append(new_row)

    # Add the untouched second part
    for _, row in df_after.iterrows():
        augmented_data.append(row)

    augmented_df = pd.DataFrame(augmented_data)

    augmented_df.to_csv(output_file, sep="\t", index=False)
    print(f"Augmented dataset saved to {output_file}")
    print(f"Original dataset size: {len(df)}")
    print(f"Augmented dataset size: {len(augmented_df)}")


if __name__ == "__main__":
    input_file = "datasets/protein_dataset.tsv"
    output_file = "datasets/protein_dataset_augmented.tsv"

    augment_dataset(
        input_file=input_file,
        output_file=output_file,
        num_substitutions=[1, 2, 3],
        num_mutations=[15, 10, 5],
    )
