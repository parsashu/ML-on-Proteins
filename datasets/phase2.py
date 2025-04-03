import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

df_train = pd.read_csv("datasets/phase2/train/X_train.csv")
df_test = pd.read_csv("datasets/phase2/test/X_test.csv")

df_train_normalized = df_train.copy()
df_test_normalized = df_test.copy()

numeric_cols = [
    "ASA",
    "pH",
    "T_(C)",
    "Tm_(C)",
    "m_(kcal/mol/M)",
    "Cm_(M)",
    "∆G_H2O_(kcal/mol)",
]

scaler = StandardScaler()

df_train_normalized = pd.DataFrame(
    scaler.fit_transform(df_train_normalized), columns=df_train.columns
)
df_test_normalized = pd.DataFrame(
    scaler.transform(df_test_normalized), columns=df_test.columns
)

df_train_normalized.to_csv("datasets/phase2/train/X_train_normalized.csv", index=False)
df_test_normalized.to_csv("datasets/phase2/test/X_test_normalized.csv", index=False)

print("Datasets normalized and saved successfully")
