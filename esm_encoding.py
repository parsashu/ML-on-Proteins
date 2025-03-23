import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import os
from tqdm import tqdm

os.makedirs("embeddings", exist_ok=True)

print("Loading ESM2 model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
model = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D", add_pooling_layer=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Using device: {device}")

print("Loading dataset...")
df = pd.read_csv("datasets/1temp_with_sequences.tsv", sep="\t")

# Filter out rows with missing sequences
df = df.dropna(subset=["Protein_Sequence"])
print(f"Dataset loaded with {len(df)} rows containing protein sequences")


# Function to generate embeddings for a sequence
def generate_embedding(sequence):
    # Skip empty sequences
    if not isinstance(sequence, str) or len(sequence) == 0:
        return None

    # Tokenize and generate embedding
    inputs = tokenizer(sequence, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Get the mean of the last hidden state as the sequence embedding
    embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]

    return embedding


# Process sequences and generate embeddings
print("Generating embeddings...")
embeddings = []
uniprot_ids = []
sequences = []

# Use tqdm for progress tracking
for idx, row in tqdm(df.iterrows(), total=len(df)):
    uniprot_id = row["UniProt_ID"]
    sequence = row["Protein_Sequence"]

    # Skip if sequence is missing
    if not isinstance(sequence, str) or len(sequence) == 0:
        continue

    embedding = generate_embedding(sequence)

    if embedding is not None:
        embeddings.append(embedding)
        uniprot_ids.append(uniprot_id)
        sequences.append(sequence)

# Create a DataFrame with the results
results_df = pd.DataFrame({"UniProt_ID": uniprot_ids, "Protein_Sequence": sequences})

# Save embeddings as a numpy array
embeddings_array = np.array(embeddings)
np.save("embeddings/protein_embeddings.npy", embeddings_array)

# Save metadata
results_df.to_csv("embeddings/protein_metadata.csv", index=False)

print(f"Completed! Generated embeddings for {len(embeddings)} proteins")
print(f"Embedding shape: {embeddings_array.shape}")
print(
    "Results saved to embeddings/protein_embeddings.npy and embeddings/protein_metadata.csv"
)
