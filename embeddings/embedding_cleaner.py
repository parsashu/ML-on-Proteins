import pandas as pd
import numpy as np

protein_df = pd.read_csv("embeddings/protein_metadata.csv")
embeddings_path = "embeddings/protein_embeddings_dim_128.npy"


embeddings = np.load(embeddings_path)

if len(embeddings) == len(protein_df):
    protein_df["Embedding"] = list(embeddings)

    sequence_column = "Protein_Sequence"
    if sequence_column not in protein_df.columns:
        potential_sequence_columns = [
            col
            for col in protein_df.columns
            if any(len(str(val)) > 30 for val in protein_df[col].head(10))
        ]
        if potential_sequence_columns:
            sequence_column = potential_sequence_columns[0]
        else:
            sequence_column = [col for col in protein_df.columns if col != "Embedding"][
                0
            ]

    print(f"Before removing duplicates: {len(protein_df)} rows")
    protein_df_unique = protein_df.drop_duplicates(
        subset=[sequence_column], keep="first"
    )
    print(f"After removing duplicates: {len(protein_df_unique)} rows")

    protein_df_unique.to_csv(
        "embeddings/protein_metadata_with_embeddings_unique.csv", index=False
    )
    print(f"Saved deduplicated data with embeddings")




import pandas as pd

protein_df = pd.read_csv("embeddings/protein_metadata_with_embeddings.csv")

sequence_column = 'Protein_Sequence'
if sequence_column not in protein_df.columns:
    potential_sequence_columns = [col for col in protein_df.columns 
                                 if any(len(str(val)) > 30 for val in protein_df[col].head(10))]
    if potential_sequence_columns:
        sequence_column = potential_sequence_columns[0]
    else:
        sequence_column = [col for col in protein_df.columns if col != 'Embedding'][0]

print(f"Before removing duplicates: {len(protein_df)} rows")
protein_df_unique = protein_df.drop_duplicates(subset=[sequence_column], keep='first')
print(f"After removing duplicates: {len(protein_df_unique)} rows")

protein_df_unique.to_csv('embeddings/protein_metadata_with_embeddings_unique.csv', index=False)
print(f"Saved deduplicated data with embeddings")



import pandas as pd

df = pd.read_csv('embeddings/protein_metadata.csv')
df_unique = df.drop_duplicates(subset=['UniProt_ID', 'Protein_Sequence'])
df_unique.to_csv('embeddings/protein_metadata_unique.csv', index=False)

print(f"Original file had {len(df)} rows")
print(f"After removing duplicates: {len(df_unique)} rows")
print(f"Removed {len(df) - len(df_unique)} duplicate entries")


