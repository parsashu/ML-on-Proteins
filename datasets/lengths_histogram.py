import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style("whitegrid")
plt.figure(figsize=(12, 6))

file_path = "datasets/sequences_filtered_1024.tsv"
df = pd.read_csv(file_path, sep="\t", header=None)

sequence_col = df.iloc[:, -1]
sequence_lengths = sequence_col.str.len()

plt.hist(sequence_lengths, bins=50, alpha=0.7, color="steelblue", edgecolor="black")

common_lengths = [256, 512, 1024, 2048]
for length in common_lengths:
    plt.axvline(
        x=length,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"{length} tokens" if length == common_lengths[0] else f"{length}",
    )

percentiles = [50, 90, 95, 99]
percentile_values = np.percentile(sequence_lengths, percentiles)

for p, val in zip(percentiles, percentile_values):
    plt.axvline(x=val, color="green", linestyle=":", alpha=0.7)
    plt.text(
        val + 10,
        plt.gca().get_ylim()[1] * 0.9,
        f"{p}th percentile: {int(val)}",
        rotation=90,
        verticalalignment="top",
    )

coverage = {}
for length in common_lengths:
    coverage[length] = (sequence_lengths <= length).mean() * 100

coverage_labels = [
    f"{length} tokens ({coverage[length]:.1f}% coverage)" for length in common_lengths
]
plt.legend(coverage_labels, loc="upper right")

plt.xlabel("Sequence Length (amino acids)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.title("Distribution of Protein Sequence Lengths", fontsize=14)

stats_text = (
    f"Min: {sequence_lengths.min()}\n"
    f"Max: {sequence_lengths.max()}\n"
    f"Mean: {sequence_lengths.mean():.1f}\n"
    f"Median: {sequence_lengths.median()}"
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

print(f"Total sequences: {len(sequence_lengths)}")
print(f"Unique sequences: {sequence_lengths.nunique()}")
print(f"Min length: {sequence_lengths.min()}")
print(f"Max length: {sequence_lengths.max()}")
print(f"Mean length: {sequence_lengths.mean():.1f}")
print(f"Median length: {sequence_lengths.median()}")
print("\nPercentiles:")
for p, val in zip(percentiles, percentile_values):
    print(f"{p}th percentile: {int(val)}")
print("\nCoverage by max length:")
for length in common_lengths:
    print(f"{length} tokens: {coverage[length]:.1f}%")
