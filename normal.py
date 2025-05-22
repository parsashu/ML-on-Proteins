import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import dump

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

X_train = pd.read_csv(X_train_path)
y_train = pd.read_csv(y_train_path)
X_val = pd.read_csv(X_val_path)
y_val = pd.read_csv(y_val_path)
X_test = pd.read_csv(X_test_path)
y_test = pd.read_csv(y_test_path)


features_to_normalize = ["ASA", "pH"]

scaler = StandardScaler()
X_train_norm = X_train.copy()
X_train_norm[features_to_normalize] = scaler.fit_transform(
    X_train[features_to_normalize]
)

X_val_norm = X_val.copy()
X_val_norm[features_to_normalize] = scaler.transform(X_val[features_to_normalize])

X_test_norm = X_test.copy()
X_test_norm[features_to_normalize] = scaler.transform(X_test[features_to_normalize])

y_scaler = StandardScaler()
y_train_norm = pd.DataFrame(
    y_scaler.fit_transform(y_train.values.reshape(-1, 1)), columns=y_train.columns
)
y_val_norm = pd.DataFrame(
    y_scaler.transform(y_val.values.reshape(-1, 1)), columns=y_val.columns
)
y_test_norm = pd.DataFrame(
    y_scaler.transform(y_test.values.reshape(-1, 1)), columns=y_test.columns
)

# Save the scalers for later use
dump(scaler, "models/feature_scaler.joblib")
dump(y_scaler, "models/target_scaler.joblib")

X_train_norm.to_csv(X_train_path, index=False)
X_val_norm.to_csv(X_val_path, index=False)
X_test_norm.to_csv(X_test_path, index=False)
y_train_norm.to_csv(y_train_path, index=False)
y_val_norm.to_csv(y_val_path, index=False)
y_test_norm.to_csv(y_test_path, index=False)
