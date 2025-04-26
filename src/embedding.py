from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

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


def embed_dataset(dataset, embeddings_file, checkpoint_interval=1000):
    """
    Generate embeddings only for sequences not already in embeddings file
    """
    df = pd.read_csv(dataset, sep="\t", low_memory=False)
    existing_df = pd.read_csv(embeddings_file)
    existing_sequences = set(existing_df["Protein_Sequence"].tolist())
    print(f"Found {len(existing_sequences)} existing sequences with embeddings")

    df = df[~df["Protein_Sequence"].isin(existing_sequences)]
    df = df.drop_duplicates(subset=["Protein_Sequence"])
    print(f"Number of new unique sequences to process: {len(df)}")

    print("Generating embeddings for new proteins...")
    embeddings = []
    sequences = []
    embedding_strings = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        sequence = row["Protein_Sequence"]

        embedding = generate_embedding(sequence)

        if embedding is not None:
            embeddings.append(embedding)
            sequences.append(sequence)
            embedding_str = str(embedding)
            embedding_strings.append(embedding_str)

        # Save embeddings to file every checkpoint_interval
        if (idx + 1) % checkpoint_interval == 0:
            new_embeddings_df = pd.DataFrame(
                {
                    "Protein_Sequence": sequences,
                    "Embedding": embedding_strings,
                }
            )
            combined_df = pd.concat([existing_df, new_embeddings_df], ignore_index=True)
            combined_df.to_csv(embeddings_file, index=False)

    # Save any remaining embeddings
    if len(sequences) > 0:
        new_embeddings_df = pd.DataFrame(
            {
                "Protein_Sequence": sequences,
                "Embedding": embedding_strings,
            }
        )
        combined_df = pd.concat([existing_df, new_embeddings_df], ignore_index=True)
        combined_df.to_csv(embeddings_file, index=False)

    print(f"Completed! Generated embeddings for {len(embeddings)} new proteins")
    print(f"Total proteins with embeddings: {len(combined_df)}")
    print(f"Results saved to {embeddings_file}")


embed_dataset(dataset, embeddings_file)
