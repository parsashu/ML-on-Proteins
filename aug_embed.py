from augmentation import augment_sequence, substitute_with_constraints
import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
model = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D", add_pooling_layer=False)
model = model.to(device)

sequence1 = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
sequence2 = "MAKVRTKDVMEQFNLELISGEEGINRPITMSDLSRPGIEIAGYFTYYPRERVQLLGK"
sequence3 = (
    "MGSSHHHHHHSSGLVPRGSHMASMTGGQQMGRGSEFDDDDKMQTIEEVLHKAIELAKVGVDSVEEAKKVLAKLLDKEK"
)


# Generate augmented sequences
augmented_sequences1 = augment_sequence(
    sequence1,
    num_substitutions=[1, 2, 3],
    num_mutations=[5, 4, 3],
    min_score=0,
    max_hydro_diff=1,
    random_seed=42,
)
augmented_sequences2 = augment_sequence(
    sequence2,
    num_substitutions=[1, 2, 3],
    num_mutations=[5, 4, 3],
    min_score=0,
    max_hydro_diff=1,
    random_seed=42,
)
augmented_sequences3 = augment_sequence(
    sequence3,
    num_substitutions=[1, 2, 3],
    num_mutations=[5, 4, 3],
    min_score=0,
    max_hydro_diff=1,
    random_seed=42,
)

all_sequences = (
    [sequence1]
    + augmented_sequences1
    + [sequence2]
    + augmented_sequences2
    + [sequence3]
    + augmented_sequences3
)

all_embeddings = []
for seq in all_sequences:
    inputs = tokenizer(seq, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).cpu()
    all_embeddings.append(embedding)

combined_embeddings = torch.cat(all_embeddings, dim=0).numpy()

pca = PCA(n_components=2)
pca_result = pca.fit_transform(combined_embeddings)

plt.figure(figsize=(12, 8))

offset1 = 0
offset2 = len(augmented_sequences1) + 1
offset3 = offset2 + len(augmented_sequences2) + 1

num_aug1 = len(augmented_sequences1)
for i in range(num_aug1 + 1):
    if i == 0:  # Original sequence
        plt.scatter(
            pca_result[offset1 + i, 0],
            pca_result[offset1 + i, 1],
            c="blue",
            marker="*",
            s=200,
            label="Original Sequence 1",
        )
    else:  # Augmented sequences
        plt.scatter(
            pca_result[offset1 + i, 0],
            pca_result[offset1 + i, 1],
            c="blue",
            alpha=0.6,
            s=40,
            label="Augmented Sequence 1" if i == 1 else None,
        )

num_aug2 = len(augmented_sequences2)
for i in range(num_aug2 + 1):
    if i == 0:  # Original sequence
        plt.scatter(
            pca_result[offset2 + i, 0],
            pca_result[offset2 + i, 1],
            c="red",
            marker="*",
            s=200,
            label="Original Sequence 2",
        )
    else:  # Augmented sequences
        plt.scatter(
            pca_result[offset2 + i, 0],
            pca_result[offset2 + i, 1],
            c="red",
            alpha=0.6,
            s=40,
            label="Augmented Sequence 2" if i == 1 else None,
        )

num_aug3 = len(augmented_sequences3)
for i in range(num_aug3 + 1):
    if i == 0:  # Original sequence
        plt.scatter(
            pca_result[offset3 + i, 0],
            pca_result[offset3 + i, 1],
            c="green",
            marker="*",
            s=200,
            label="Original Sequence 3",
        )
    else:  # Augmented sequences
        plt.scatter(
            pca_result[offset3 + i, 0],
            pca_result[offset3 + i, 1],
            c="green",
            alpha=0.6,
            s=40,
            label="Augmented Sequence 3" if i == 1 else None,
        )

plt.title("PCA of Protein Sequence Embeddings")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True)
plt.show()
