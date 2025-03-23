import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import os
from tqdm import tqdm

os.makedirs("embeddings", exist_ok=True)

print("Loading protein language model and tokenizer...")
model_name = "facebook/esm2_t6_8M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModel.from_pretrained(model_name)


class ESMWithDimReduction(torch.nn.Module):
    def __init__(self, base_model, output_dim=128):
        super().__init__()
        self.base_model = base_model
        self.dim_reduction = torch.nn.Linear(base_model.config.hidden_size, output_dim)

    def forward(self, **inputs):
        outputs = self.base_model(**inputs)
        reduced_embeddings = self.dim_reduction(outputs.last_hidden_state)
        outputs.last_hidden_state = reduced_embeddings
        return outputs


model = ESMWithDimReduction(base_model, output_dim=128)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Using device: {device}")

print("Loading dataset...")
df = pd.read_csv("datasets/sequences_filtered_1024.tsv", sep="\t")
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

for idx, row in tqdm(df.iterrows(), total=len(df)):
    uniprot_id = row["UniProt_ID"]
    sequence = row["Protein_Sequence"]

    embedding = generate_embedding(sequence)

    if embedding is not None:
        embeddings.append(embedding)
        uniprot_ids.append(uniprot_id)
        sequences.append(sequence)

results_df = pd.DataFrame({"UniProt_ID": uniprot_ids, "Protein_Sequence": sequences})

embeddings_array = np.array(embeddings)
embedding_dim = embeddings_array.shape[1]

np.save(f"embeddings/protein_embeddings_dim_{embedding_dim}.npy", embeddings_array)
results_df.to_csv("embeddings/protein_metadata.csv", index=False)

print(f"Completed! Generated embeddings for {len(embeddings)} proteins")
print(f"Embedding dimension: {embedding_dim}")
print(f"Embeddings shape: {embeddings_array.shape}")
print(
    f"Results saved to embeddings/protein_embeddings_dim_{embedding_dim}.npy and embeddings/protein_metadata.csv"
)
