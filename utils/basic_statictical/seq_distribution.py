import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def sequence_length_distribution(file_path, max_length=15000):
    """Plot the distribution of unique protein sequence lengths."""
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))

    df = pd.read_csv(file_path, sep="\t", low_memory=False)
    unique_sequences = df["Protein_Sequence"].unique()
    sequence_lengths = [len(seq) for seq in unique_sequences]

    plt.hist(sequence_lengths, bins=50, alpha=0.7, color="steelblue", edgecolor="black")

    if max_length is not None:
        plt.axvline(
            x=max_length,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"{max_length} tokens",
        )

    coverage = {}
    if max_length is not None:
        coverage[max_length] = (np.array(sequence_lengths) <= max_length).mean() * 100
        coverage_label = f"{max_length} tokens ({coverage[max_length]:.1f}% coverage)"
        plt.legend([coverage_label], loc="upper right")

    plt.xlabel("Sequence Length (amino acids)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("Distribution of Unique Protein Sequence Lengths", fontsize=14)

    stats_text = (
        f"Min: {min(sequence_lengths)}\n"
        f"Max: {max(sequence_lengths)}\n"
        f"Mean: {np.mean(sequence_lengths):.1f}\n"
        f"Median: {np.median(sequence_lengths)}"
    )
    plt.text(
        0.02,
        0.95,
        stats_text,
        transform=plt.gca().transAxes,
        bbox=dict(facecolor="white", alpha=0.8),
    )

    plt.xlim(left=0)
    plt.tight_layout()
    plt.show()
