from transformers import AutoTokenizer, AutoModel
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
model = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D", add_pooling_layer=False)

sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
print(len(sequence))

inputs = tokenizer(sequence, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

embeddings = outputs.last_hidden_state.mean(dim=1)

print(f"Embedding shape: {embeddings.shape}")

plt.figure(figsize=(12, 4))
sns.heatmap(embeddings.numpy(), cmap="viridis")
plt.title("Protein Embedding Vector Heatmap")
plt.xlabel("Embedding Dimension")
plt.ylabel("Sequence")
plt.show()

token_embeddings = outputs.last_hidden_state[0]
plt.figure(figsize=(12, 8))
sns.heatmap(token_embeddings.numpy()[:, :20], cmap="viridis")
plt.title("Per-token Embeddings (first 20 dimensions)")
plt.xlabel("Embedding Dimension")
plt.ylabel("Token Position")
plt.show()

pca = PCA(n_components=2)
token_embeddings_2d = pca.fit_transform(token_embeddings.numpy())

plt.figure(figsize=(10, 8))
plt.scatter(
    token_embeddings_2d[:, 0],
    token_embeddings_2d[:, 1],
    c=np.arange(token_embeddings_2d.shape[0]),
    cmap="viridis",
)
plt.colorbar(label="Token Position")
plt.title("PCA of Token Embeddings")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# Unembedding the sequence
print("\nUnembedding the sequence:")
tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
amino_acids = [token.replace("<", "").replace(">", "") for token in tokens]
amino_acids = [aa for aa in amino_acids if aa not in ["cls", "pad", "eos", "mask"]]
unembedded_sequence = "".join(amino_acids)
print("Original sequence:", sequence)
print("Unembedded sequence:", unembedded_sequence)
print("Equal:", sequence == unembedded_sequence)
