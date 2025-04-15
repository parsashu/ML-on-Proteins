import torch
import pandas as pd
from tqdm import tqdm
import os
from transformers import AutoTokenizer, AutoModel
import numpy as np

dataset = "datasets/protein_dataset.tsv"
embeddings_file = "datasets/sequence_embeddings.csv"
sequences_dataset = "datasets/raw/sequences_dataset.csv"

# Load and save all unique sequences
df = pd.read_csv(dataset, sep="\t", low_memory=False)
print(f"Number of rows before removing duplicates: {len(df)}")

df = df[["UniProt_ID", "Protein_Sequence"]]
df = df.drop_duplicates(subset=["Protein_Sequence"], keep="first")
print(f"Number of rows after removing duplicates: {len(df)}")

df.to_csv(sequences_dataset, index=False)
print(f"Unique protein data saved to {sequences_dataset}")


# Generate embeddings only for sequences not already in embeddings file
print("Loading protein language model and tokenizer...")
model_name = "facebook/esm2_t6_8M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Using device: {device}")

# Load existing embeddings if file exists
if os.path.exists(embeddings_file):
    existing_df = pd.read_csv(embeddings_file)
    existing_sequences = set(existing_df["Protein_Sequence"].tolist())
    print(f"Found {len(existing_sequences)} existing sequences with embeddings")
else:
    existing_df = pd.DataFrame(columns=["UniProt_ID", "Protein_Sequence", "Embedding"])
    existing_sequences = set()
    print("No existing embeddings found. Will create new file.")

# Filter to only process sequences not already in embeddings file
df = df[~df["Protein_Sequence"].isin(existing_sequences)]
print(f"Number of new sequences to process: {len(df)}")

# Filter out sequences longer than max_length
max_length = 15000
original_count = len(df)
df = df[df["Protein_Sequence"].str.len() <= max_length]
filtered_count = original_count - len(df)
print(f"Filtered out {filtered_count} sequences longer than {max_length} characters")
print(f"Remaining sequences to process: {len(df)}")

checkpoint_interval = 100
checkpoint_file = "embedding_checkpoint.csv"


def load_checkpoint():
    if os.path.exists(checkpoint_file):
        print(f"Loading checkpoint from {checkpoint_file}")
        checkpoint_df = pd.read_csv(checkpoint_file)
        embeddings = []
        for emb_str in checkpoint_df["Embedding"]:
            emb_array = np.fromstring(emb_str.strip("[]"), sep=" ")
            embeddings.append(emb_array)

        return {
            "embeddings": embeddings,
            "uniprot_ids": checkpoint_df["UniProt_ID"].tolist(),
            "sequences": checkpoint_df["Protein_Sequence"].tolist(),
            "embedding_strings": checkpoint_df["Embedding"].tolist(),
            "current_idx": len(checkpoint_df) - 1,
        }
    return None


def save_checkpoint(embeddings, uniprot_ids, sequences, embedding_strings, current_idx):
    checkpoint_df = pd.DataFrame(
        {
            "UniProt_ID": uniprot_ids,
            "Protein_Sequence": sequences,
            "Embedding": embedding_strings,
        }
    )
    checkpoint_df.to_csv(checkpoint_file, index=False)


def generate_embedding(sequence, normalize=True):
    if not isinstance(sequence, str) or len(sequence) == 0:
        return None

    inputs = tokenizer(sequence, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]

    # L2 normalization (unit vector)
    if normalize:
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

    return embedding


print("Generating embeddings for new proteins...")
embeddings = []
uniprot_ids = []
sequences = []
embedding_strings = []
start_idx = 0

checkpoint = load_checkpoint()
if checkpoint:
    embeddings = checkpoint["embeddings"]
    uniprot_ids = checkpoint["uniprot_ids"]
    sequences = checkpoint["sequences"]
    embedding_strings = checkpoint["embedding_strings"]
    start_idx = checkpoint["current_idx"] + 1
    print(f"Resuming from index {start_idx}")

for idx, row in tqdm(df.iterrows(), total=len(df)):
    uniprot_id = row["UniProt_ID"]
    sequence = row["Protein_Sequence"]

    embedding = generate_embedding(sequence)

    if embedding is not None:
        embeddings.append(embedding)
        uniprot_ids.append(uniprot_id)
        sequences.append(sequence)
        embedding_str = str(embedding)
        embedding_strings.append(embedding_str)

    if (idx + 1) % checkpoint_interval == 0:
        save_checkpoint(embeddings, uniprot_ids, sequences, embedding_strings, idx)

results_df = pd.DataFrame(
    {
        "UniProt_ID": uniprot_ids,
        "Protein_Sequence": sequences,
        "Embedding": embedding_strings,
    }
)

# Combine new embeddings with existing ones
combined_df = pd.concat([existing_df, results_df], ignore_index=True)
combined_df.to_csv(embeddings_file, index=False)

if os.path.exists(checkpoint_file):
    os.remove(checkpoint_file)

print(f"Completed! Generated embeddings for {len(embeddings)} new proteins")
print(f"Total proteins with embeddings: {len(combined_df)}")
print(f"Embedding dimension: {len(embeddings[0]) if embeddings else 0}")
print(f"Results saved to {embeddings_file}")
