import pandas as pd

file_path = 'datasets/1temp_with_sequences_filtered.tsv'
df = pd.read_csv(file_path, sep='\t', header=None)

sequence_col = df.iloc[:, -1]

sequence_lengths = sequence_col.str.len()

max_length = 1024
filtered_df = df[sequence_lengths <= max_length]

output_file = 'datasets/sequences_filtered_1024.tsv'
filtered_df.to_csv(output_file, sep='\t', header=False, index=False)

total_sequences = len(df)
filtered_sequences = len(filtered_df)
removed_sequences = total_sequences - filtered_sequences

print(f"Total sequences: {total_sequences}")
print(f"Sequences within length limit ({max_length}): {filtered_sequences}")
print(f"Sequences removed: {removed_sequences} ({removed_sequences/total_sequences*100:.2f}%)")
print(f"Filtered data saved to: {output_file}")