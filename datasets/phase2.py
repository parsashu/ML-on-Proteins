import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv("datasets/features.csv")

features = [
    "ASA",
    "pH",
    "T_(C)",
    "Coil",
    "Helix",
    "Sheet",
    "Turn",
    "Tm_(C)",
    "m_(kcal/mol/M)",
    "Cm_(M)",
    "∆G_H2O_(kcal/mol)",
]

# Generate a list of embedding feature names
embed_features = [f"embed_{i}" for i in range(320)]

features_df = df[features].copy()
features_df = features_df.fillna(-1000)

labels_df = df[embed_features].copy()
labels_df = labels_df.fillna(0)

train_features, test_features, train_labels, test_labels = train_test_split(
    features_df, labels_df, test_size=0.1, random_state=42
)

train_features.to_csv("datasets/phase2/train/X_train.csv", index=False)
test_features.to_csv("datasets/phase2/test/X_test.csv", index=False)
train_labels.to_csv("datasets/phase2/train/y_train.csv", index=False)
train_labels.to_csv("datasets/phase2/train/y_train.csv.gz", index=False, compression="gzip")
test_labels.to_csv("datasets/phase2/test/y_test.csv", index=False)
test_labels.to_csv("datasets/phase2/test/y_test.csv.gz", index=False, compression="gzip")

print(
    "Features and labels split and saved to datasets/phase2/train and datasets/phase2/test directories"
)
