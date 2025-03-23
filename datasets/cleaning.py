import pandas as pd

input_file = "datasets/1temp_with_sequences.tsv"
output_file = "datasets/1temp_with_sequences_filtered.tsv"

df = pd.read_csv(input_file, sep="\t")
total_rows = len(df)

df_filtered = df.dropna(subset=["Protein_Sequence"])
df_filtered = df_filtered[df_filtered["Protein_Sequence"].str.strip() != ""]

filtered_rows = len(df_filtered)
removed_rows = total_rows - filtered_rows

df_filtered.to_csv(output_file, sep="\t", index=False)

print(f"Total rows in original file: {total_rows}")
print(f"Rows removed (no sequence): {removed_rows}")
print(f"Rows remaining: {filtered_rows}")
print(f"Filtered data saved to {output_file}")
