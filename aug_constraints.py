import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from augmentation import augment_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
model = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D", add_pooling_layer=False)
model = model.to(device)

sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"

# Generate augmented sequences with different constraints
augmented_sequences1 = augment_sequence(
    sequence,
    num_substitutions=[1, 2, 3],
    num_mutations=[5, 4, 3],
    min_score=0,
    max_hydro_diff=1,
    random_seed=42,
)

augmented_sequences2 = augment_sequence(
    sequence,
    num_substitutions=[1, 2, 3],
    num_mutations=[5, 4, 3],
    min_score=1,
    max_hydro_diff=0.5,
    random_seed=42,
)

# Get embeddings for original sequence
original_inputs = tokenizer(sequence, return_tensors="pt").to(device)
with torch.no_grad():
    original_outputs = model(**original_inputs)
original_embedding = original_outputs.last_hidden_state.mean(dim=1).cpu()

# Get embeddings for first set of augmented sequences
embeddings1 = []
for seq in augmented_sequences1:
    inputs = tokenizer(seq, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).cpu()
    embeddings1.append(embedding)

# Get embeddings for second set of augmented sequences
embeddings2 = []
for seq in augmented_sequences2:
    inputs = tokenizer(seq, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).cpu()
    embeddings2.append(embedding)

# Combine embeddings for PCA
combined_embeddings = torch.cat(
    [original_embedding] + embeddings1 + embeddings2, dim=0
).numpy()

pca = PCA(n_components=2)
pca_result = pca.fit_transform(combined_embeddings)

plt.figure(figsize=(12, 8))
plt.scatter(
    pca_result[0, 0],
    pca_result[0, 1],
    c="black",
    marker="*",
    s=200,
    label="Original Sequence",
)

offset1 = 1
offset2 = len(augmented_sequences1) + 1

for i in range(len(augmented_sequences1)):
    plt.scatter(
        pca_result[offset1 + i, 0],
        pca_result[offset1 + i, 1],
        c="blue",
        alpha=0.6,
        s=40,
        label="Augmented (min_score=0, max_hydro_diff=1)" if i == 0 else None,
    )

for i in range(len(augmented_sequences2)):
    plt.scatter(
        pca_result[offset2 + i, 0],
        pca_result[offset2 + i, 1],
        c="red",
        alpha=0.6,
        s=40,
        label="Augmented (min_score=1, max_hydro_diff=0.5)" if i == 0 else None,
    )

plt.title("PCA of Protein Sequence Embeddings with Different Constraints")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True)
plt.show()


# Calculate L2 norm differences between original and augmented embeddings
def difference_from_original(embeddings, original_embedding):
    differences = []
    for embedding in embeddings:
        diff = embedding.numpy() - original_embedding.numpy()
        diff = np.linalg.norm(diff, axis=1)
        differences.append(diff)
    return sum(differences)[0] / len(differences)


loss1 = difference_from_original(embeddings1, original_embedding)
loss2 = difference_from_original(embeddings2, original_embedding)

print(loss1)
print(loss2)
