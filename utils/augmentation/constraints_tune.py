import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import numpy as np
from augment_functions import augment_sequence
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
model = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D", add_pooling_layer=False)
model = model.to(device)

sequence = "MAKVRTKDVMEQFNLELISGEEGINRPITMSDLSRPGIEIAGYFTYYPRERVQLLGKTELSFFEQLPEEEKKQRMDSLCTDVTPAIILSRDMPIPQELIDASEKNGVPVLRSPLKTTRLSSRLTNFLESRLAPTTAIHGVLVDIYGVGVLITGKSGVGKSETALELVKRGHRLVADDCVEIRQEDQDTLVGNAPELIEHLLEIRGLGIINVMTLFGAGAVRSNKRITIVMNLELWEQGKQYDRLGLEEETMKIIDTEITKLTIPVRPGRNLAVIIEVAAMNFRLKRMGLNAAEQFTNKLADVIEDGEQEE"

min_scores = [0, 1, 2, 3]
max_hydro_diffs = [4, 3.5, 3, 2.5, 2, 1.5, 1, 0.5]

# Get embeddings for original sequence
original_inputs = tokenizer(sequence, return_tensors="pt").to(device)
with torch.no_grad():
    original_outputs = model(**original_inputs)
original_embedding = original_outputs.last_hidden_state.mean(dim=1).cpu()


# Calculate L2 norm differences between original and augmented embeddings
def difference_from_original(embeddings, original_embedding):
    differences = []
    for embedding in embeddings:
        diff = embedding.numpy() - original_embedding.numpy()
        diff = np.linalg.norm(diff, axis=1)
        differences.append(diff)
    return sum(differences)[0] / len(differences)


# Initialize arrays to store results for ensemble averaging
num_runs = 10
all_loss_matrices = np.zeros((num_runs, len(min_scores), len(max_hydro_diffs)))

# Run experiments with different random seeds
for run in range(num_runs):
    loss_matrix = np.zeros((len(min_scores), len(max_hydro_diffs)))

    for i, min_score in enumerate(min_scores):
        for j, max_hydro_diff in enumerate(max_hydro_diffs):
            augmented_sequences = augment_sequence(
                sequence,
                num_substitutions=[1, 2, 3],
                num_mutations=[10, 0, 0],
                min_score=min_score,
                max_hydro_diff=max_hydro_diff,
            )

            embeddings = []
            for seq in augmented_sequences:
                inputs = tokenizer(seq, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).cpu()
                embeddings.append(embedding)

            loss = difference_from_original(embeddings, original_embedding)
            loss_matrix[i, j] = loss

    all_loss_matrices[run] = loss_matrix
# Calculate mean across runs
mean_loss_matrix = np.mean(all_loss_matrices, axis=0)

# Create single figure
plt.figure(figsize=(8, 6))

# Plot mean heatmap
sns.heatmap(
    mean_loss_matrix,
    annot=True,
    fmt=".5f",
    cmap="YlOrRd",
    xticklabels=max_hydro_diffs,
    yticklabels=min_scores,
    cbar_kws={"label": "Average L2 Distance from Original"},
)
plt.title("Mean Loss Heatmap Across 10 Runs")
plt.xlabel("Max Hydrophobicity Difference")
plt.ylabel("Min Score")
plt.show()

# Fix min_score to 3 and plot loss vs max_hydro_diff with ensemble averaging
fixed_min_score = 3
num_runs = 20
max_hydro_diffs = np.linspace(0.5, 6, 10)
all_losses = np.zeros((num_runs, len(max_hydro_diffs)))

# Ensemble averaging
for run in range(num_runs):
    losses = []
    for i, max_hydro_diff in enumerate(max_hydro_diffs):
        augmented_sequences = augment_sequence(
            sequence,
            num_substitutions=[1, 2, 3],
            num_mutations=[10, 0, 0],
            min_score=fixed_min_score,
            max_hydro_diff=max_hydro_diff,
        )

        embeddings = []
        for seq in augmented_sequences:
            inputs = tokenizer(seq, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).cpu()
            embeddings.append(embedding)

        loss = difference_from_original(embeddings, original_embedding)
        losses.append(loss)

    all_losses[run] = losses
mean_losses = np.mean(all_losses, axis=0)

plt.figure(figsize=(10, 6))
plt.plot(max_hydro_diffs, mean_losses, marker="o", linestyle="-", label="Mean Loss")
plt.title(
    f"Loss vs Max Hydrophobicity Difference (Min Score = {fixed_min_score}, Ensemble Size = {num_runs})"
)
plt.xlabel("Max Hydrophobicity Difference")
plt.ylabel("L2 Distance from Original")
plt.grid(True)
plt.legend()
plt.show()
