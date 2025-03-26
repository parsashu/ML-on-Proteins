import pandas as pd
import os



df = pd.read_csv("datasets/protein_dataset.tsv", sep="\t")
print(f"Number of rows before removing duplicates: {len(df)}")

# Keep only the required columns
df = df[["UniProt_ID", "Protein_Sequence"]]

# Remove duplicates based on UniProt_ID
df = df.drop_duplicates(subset=["UniProt_ID"], keep="first")
print(f"Number of rows after removing duplicates: {len(df)}")

# Create embeddings directory if it doesn't exist
os.makedirs("embeddings", exist_ok=True)

# Save the unique data
df.to_csv("embeddings/protein_metadata_unique.csv", index=False)
print("Unique protein data saved to embeddings/protein_metadata_unique.csv")

