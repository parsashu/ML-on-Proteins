from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

dataset = "datasets/train_dataset_augmented.tsv"
embeddings_file = "datasets/sequence_embeddings.csv"


print("Loading protein language model and tokenizer...")
model_name = "facebook/esm2_t6_8M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Using device: {device}")


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


def embed_dataset(
    dataset,
    embeddings_file,
    batch_size=100,
    max_sequence_length=6000,
):
    """
    Generate embeddings only for sequences not already in embeddings file
    Process sequences in smaller batches to manage memory usage
    Skip sequences longer than max_sequence_length
    """
    df = pd.read_csv(dataset, sep="\t", low_memory=False)
    existing_df = pd.read_csv(embeddings_file)
    existing_sequences = set(existing_df["Protein_Sequence"].tolist())
    print(f"Found {len(existing_sequences)} existing sequences with embeddings")

    df = df[~df["Protein_Sequence"].isin(existing_sequences)]
    df = df.drop_duplicates(subset=["Protein_Sequence"])
    df = df[df["Protein_Sequence"].str.len() <= max_sequence_length]
    print(f"Number of new unique sequences to process: {len(df)}")

    print("Generating embeddings for new proteins...")
    i = 0
    total_embeddings = 0
    batch_data = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing sequences"):
        i += 1
        sequence = row["Protein_Sequence"]
        embedding = generate_embedding(sequence)

        if embedding is not None:
            batch_data.append(
                {"Protein_Sequence": sequence, "Embedding": str(embedding)}
            )
            total_embeddings += 1

        # Save embeddings to file every batch_size
        if (i + 1) % batch_size == 0 or i == len(df):
            if batch_data:
                new_embeddings_df = pd.DataFrame(batch_data)
                new_embeddings_df.to_csv(
                    embeddings_file,
                    mode="a",
                    header=not os.path.exists(embeddings_file),
                    index=False,
                )
                batch_data = []

            # Clear memory after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"Completed! Generated embeddings for {total_embeddings} new proteins")
    print(f"Results saved to {embeddings_file}")


embed_dataset(
    dataset,
    embeddings_file,
    batch_size=100,
    max_sequence_length=100,
)
