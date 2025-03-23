import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import os
from tqdm import tqdm

os.makedirs("embeddings", exist_ok=True)

print("Loading protein language model and tokenizer...")
model_name = "facebook/esm2_t6_8M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Using device: {device}")

print("Loading dataset...")
df = pd.read_csv("embeddings/protein_metadata_unique.csv")
print(f"Dataset loaded with {len(df)} rows containing protein sequences")


def generate_embedding(sequence):
    if not isinstance(sequence, str) or len(sequence) == 0:
        return None

    inputs = tokenizer(sequence, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
    return embedding


print("Generating embeddings for all proteins...")
embeddings = []
uniprot_ids = []
sequences = []
embedding_strings = []

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

results_df = pd.DataFrame(
    {
        "UniProt_ID": uniprot_ids,
        "Protein_Sequence": sequences,
        "Embedding": embedding_strings,
    }
)

results_df.to_csv("embeddings/protein_metadata_with_embeddings128.csv", index=False)

print(f"Completed! Generated embeddings for {len(embeddings)} proteins")
print(f"Embedding dimension: {len(embedding) if embeddings else 0}")
print(f"Results saved to embeddings/protein_metadata_with_embeddings128.csv")
