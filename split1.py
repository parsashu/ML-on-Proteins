import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("datasets/features.csv")

X_features = ["Coil", "Helix", "Turn", "Sheet", "ASA", "pH", "T_(C)"]
embedding_features = [f"embed_{i}" for i in range(320)]
X_features.extend(embedding_features)

# Model 1: Tm
df1 = df.dropna(subset=["Tm_(C)"])
X1 = df1[X_features]
y1 = df1["Tm_(C)"]

# Model 2: m
df2 = df.dropna(subset=["m_(kcal/mol/M)"])
X2 = df2[X_features]
y2 = df2["m_(kcal/mol/M)"]

# Model 3: Cm
df3 = df.dropna(subset=["Cm_(M)"])
X3 = df3[X_features]
y3 = df3["Cm_(M)"]

# Model 4: ΔG
df4 = df.dropna(subset=["∆G_H2O_(kcal/mol)"])
X4 = df4[X_features]
y4 = df4["∆G_H2O_(kcal/mol)"]

X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1, y1, test_size=0.2, random_state=42
)
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, random_state=42
)
X3_train, X3_test, y3_train, y3_test = train_test_split(
    X3, y3, test_size=0.2, random_state=42
)
X4_train, X4_test, y4_train, y4_test = train_test_split(
    X4, y4, test_size=0.2, random_state=42
)

# Model 1
pd.DataFrame(X1_train).to_csv("datasets/phase1/train/X1_train.csv", index=False)
pd.DataFrame(X1_test).to_csv("datasets/phase1/test/X1_test.csv", index=False)
pd.DataFrame(y1_train).to_csv("datasets/phase1/train/y1_train.csv", index=False)
pd.DataFrame(y1_test).to_csv("datasets/phase1/test/y1_test.csv", index=False)

# Model 2
pd.DataFrame(X2_train).to_csv("datasets/phase1/train/X2_train.csv", index=False)
pd.DataFrame(X2_test).to_csv("datasets/phase1/test/X2_test.csv", index=False)
pd.DataFrame(y2_train).to_csv("datasets/phase1/train/y2_train.csv", index=False)
pd.DataFrame(y2_test).to_csv("datasets/phase1/test/y2_test.csv", index=False)

# Model 3
pd.DataFrame(X3_train).to_csv("datasets/phase1/train/X3_train.csv", index=False)
pd.DataFrame(X3_test).to_csv("datasets/phase1/test/X3_test.csv", index=False)
pd.DataFrame(y3_train).to_csv("datasets/phase1/train/y3_train.csv", index=False)
pd.DataFrame(y3_test).to_csv("datasets/phase1/test/y3_test.csv", index=False)

# Model 4
pd.DataFrame(X4_train).to_csv("datasets/phase1/train/X4_train.csv", index=False)
pd.DataFrame(X4_test).to_csv("datasets/phase1/test/X4_test.csv", index=False)
pd.DataFrame(y4_train).to_csv("datasets/phase1/train/y4_train.csv", index=False)
pd.DataFrame(y4_test).to_csv("datasets/phase1/test/y4_test.csv", index=False)

print("\nDataset sizes after removing nulls:")
print(f"Model 1 (Tm): {X1.shape[0]} samples")
print(f"Model 2 (m): {X2.shape[0]} samples")
print(f"Model 3 (Cm): {X3.shape[0]} samples")
print(f"Model 4 (ΔG): {X4.shape[0]} samples")

print("\nTrain/Test shapes:")
print(f"Model 1 - Train: {X1_train.shape}, Test: {X1_test.shape}")
print(f"Model 2 - Train: {X2_train.shape}, Test: {X2_test.shape}")
print(f"Model 3 - Train: {X3_train.shape}, Test: {X3_test.shape}")
print(f"Model 4 - Train: {X4_train.shape}, Test: {X4_test.shape}")

print(
    "\nFeatures and labels split and saved to phase1/train and phase1/test directories"
)
