import pandas as pd
from sklearn.preprocessing import StandardScaler

seq_extracted = "datasets/raw/merged_dataset.tsv"
dataset = "datasets/protein_dataset.tsv"
Tm_dataset = "datasets/Tm_dataset.tsv"
train_path = "datasets/train_dataset.tsv"
val_path = "datasets/val_dataset.tsv"
test_path = "datasets/test_dataset.tsv"
train_augmented = "datasets/train_dataset_augmented.tsv"
embeddings_train = "datasets/embeddings_train.csv"
embeddings_val = "datasets/embeddings_val.csv"
embeddings_test = "datasets/embeddings_test.csv"
train_processed = "datasets/train_processed.csv"
val_processed = "datasets/val_processed.csv"
test_processed = "datasets/test_processed.csv"

X_train_path = "../datasets/inputs/train/X_train.csv"
y_train_path = "../datasets/inputs/train/y_train.csv"
X_val_path = "../datasets/inputs/val/X_val.csv"
y_val_path = "../datasets/inputs/val/y_val.csv"
X_test_path = "../datasets/inputs/test/X_test.csv"
y_test_path = "../datasets/inputs/test/y_test.csv"

X_train_norm = pd.read_csv(X_train_path)
y_train_norm = pd.read_csv(y_train_path)
X_val_norm = pd.read_csv(X_val_path)
y_val_norm = pd.read_csv(y_val_path)
X_test_norm = pd.read_csv(X_test_path)
y_test_norm = pd.read_csv(y_test_path)


random_state = 42

train_indices = X_train_norm.index
shuffled_indices = pd.Series(train_indices).sample(frac=1, random_state=random_state)
X_train_shuffled = X_train_norm.loc[shuffled_indices].reset_index(drop=True)
y_train_shuffled = y_train_norm.loc[shuffled_indices].reset_index(drop=True)

val_indices = X_val_norm.index
shuffled_indices = pd.Series(val_indices).sample(frac=1, random_state=random_state)
X_val_shuffled = X_val_norm.loc[shuffled_indices].reset_index(drop=True)
y_val_shuffled = y_val_norm.loc[shuffled_indices].reset_index(drop=True)

test_indices = X_test_norm.index
shuffled_indices = pd.Series(test_indices).sample(frac=1, random_state=random_state)
X_test_shuffled = X_test_norm.loc[shuffled_indices].reset_index(drop=True)
y_test_shuffled = y_test_norm.loc[shuffled_indices].reset_index(drop=True)

X_train_shuffled.to_csv(X_train_path, index=False)
X_val_shuffled.to_csv(X_val_path, index=False)
X_test_shuffled.to_csv(X_test_path, index=False)
y_train_shuffled.to_csv(y_train_path, index=False)
y_val_shuffled.to_csv(y_val_path, index=False)
y_test_shuffled.to_csv(y_test_path, index=False)

