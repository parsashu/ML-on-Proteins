import os
import sys
import pandas as pd
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from augment_functions import augment_sequence


min_score = 2
max_hydro_diff = 1


def augment_dataset(
    input_file, output_file, num_substitutions=[1, 2, 3], num_mutations=[5, 4, 3]
):
    """Augment protein sequences in the dataset while preserving other features."""
    df = pd.read_csv(input_file, sep="\t", low_memory=False)
    total_failed_count = 0

    augmented_data = []
    for i, (_, row) in tqdm(
        enumerate(df.iterrows()), total=len(df), desc="Augmenting sequences"
    ):
        augmented_data.append(row)
        original_seq = row["Protein_Sequence"]

        # Generate augmented sequences
        augmented_seqs, failed_count = augment_sequence(
            original_seq,
            num_substitutions=num_substitutions,
            num_mutations=num_mutations,
            min_score=min_score,
            max_hydro_diff=max_hydro_diff,
            retries=20,
            print_failures=False,
            random_seed=42,
        )
        total_failed_count += failed_count

        for aug_seq in augmented_seqs:
            new_row = row.copy()
            new_row["Protein_Sequence"] = aug_seq
            augmented_data.append(new_row)

    augmented_df = pd.DataFrame(augmented_data)

    augmented_df.to_csv(output_file, sep="\t", index=False)
    print(f"Augmented dataset saved to {output_file}")
    print(f"Original dataset size: {len(df)}")
    print(f"Augmented dataset size: {len(augmented_df)}")
    print(f"Total failed count: {total_failed_count}")


train_file = "datasets/train_dataset.tsv"
train_augmented = "datasets/train_dataset_augmented.tsv"

augment_dataset(
    input_file=train_file,
    output_file=train_augmented,
    num_substitutions=[1, 2, 3],
    num_mutations=[15, 10, 5],
)
