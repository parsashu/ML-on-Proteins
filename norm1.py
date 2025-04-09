import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Model 1
X1_train = pd.read_csv("datasets/phase1/train/X1_train.csv")
X1_test = pd.read_csv("datasets/phase1/test/X1_test.csv")
y1_train = pd.read_csv("datasets/phase1/train/y1_train.csv")
y1_test = pd.read_csv("datasets/phase1/test/y1_test.csv")

# Model 2
X2_train = pd.read_csv("datasets/phase1/train/X2_train.csv")
X2_test = pd.read_csv("datasets/phase1/test/X2_test.csv")
y2_train = pd.read_csv("datasets/phase1/train/y2_train.csv")
y2_test = pd.read_csv("datasets/phase1/test/y2_test.csv")

# Model 3
X3_train = pd.read_csv("datasets/phase1/train/X3_train.csv")
X3_test = pd.read_csv("datasets/phase1/test/X3_test.csv")
y3_train = pd.read_csv("datasets/phase1/train/y3_train.csv")
y3_test = pd.read_csv("datasets/phase1/test/y3_test.csv")

# Model 4
X4_train = pd.read_csv("datasets/phase1/train/X4_train.csv")
X4_test = pd.read_csv("datasets/phase1/test/X4_test.csv")
y4_train = pd.read_csv("datasets/phase1/train/y4_train.csv")
y4_test = pd.read_csv("datasets/phase1/test/y4_test.csv")

features_to_normalize = ["ASA", "pH", "T_(C)"]

for X_train, X_test in [
    (X1_train, X1_test),
    (X2_train, X2_test),
    (X3_train, X3_test),
    (X4_train, X4_test),
]:
    scaler = StandardScaler()
    X_train[features_to_normalize] = scaler.fit_transform(
        X_train[features_to_normalize]
    )
    X_test[features_to_normalize] = scaler.transform(X_test[features_to_normalize])

# Normalize y values
y_scaler1 = StandardScaler()
y_scaler2 = StandardScaler()
y_scaler3 = StandardScaler()
y_scaler4 = StandardScaler()

# Normalize y1 (Tm_(C))
y1_train_normalized = pd.DataFrame(
    y_scaler1.fit_transform(y1_train), columns=y1_train.columns
)
y1_test_normalized = pd.DataFrame(y_scaler1.transform(y1_test), columns=y1_test.columns)

# Normalize y2 (m_(kcal/mol/M))
y2_train_normalized = pd.DataFrame(
    y_scaler2.fit_transform(y2_train), columns=y2_train.columns
)
y2_test_normalized = pd.DataFrame(y_scaler2.transform(y2_test), columns=y2_test.columns)

# Normalize y3 (Cm_(M))
y3_train_normalized = pd.DataFrame(
    y_scaler3.fit_transform(y3_train), columns=y3_train.columns
)
y3_test_normalized = pd.DataFrame(y_scaler3.transform(y3_test), columns=y3_test.columns)

# Normalize y4 (∆G_H2O_(kcal/mol))
y4_train_normalized = pd.DataFrame(
    y_scaler4.fit_transform(y4_train), columns=y4_train.columns
)
y4_test_normalized = pd.DataFrame(y_scaler4.transform(y4_test), columns=y4_test.columns)

X1_train.to_csv("datasets/phase1/train/X1_train.csv", index=False)
X1_test.to_csv("datasets/phase1/test/X1_test.csv", index=False)
X2_train.to_csv("datasets/phase1/train/X2_train.csv", index=False)
X2_test.to_csv("datasets/phase1/test/X2_test.csv", index=False)
X3_train.to_csv("datasets/phase1/train/X3_train.csv", index=False)
X3_test.to_csv("datasets/phase1/test/X3_test.csv", index=False)
X4_train.to_csv("datasets/phase1/train/X4_train.csv", index=False)
X4_test.to_csv("datasets/phase1/test/X4_test.csv", index=False)

y1_train_normalized.to_csv("datasets/phase1/train/y1_train.csv", index=False)
y1_test_normalized.to_csv("datasets/phase1/test/y1_test.csv", index=False)
y2_train_normalized.to_csv("datasets/phase1/train/y2_train.csv", index=False)
y2_test_normalized.to_csv("datasets/phase1/test/y2_test.csv", index=False)
y3_train_normalized.to_csv("datasets/phase1/train/y3_train.csv", index=False)
y3_test_normalized.to_csv("datasets/phase1/test/y3_test.csv", index=False)
y4_train_normalized.to_csv("datasets/phase1/train/y4_train.csv", index=False)
y4_test_normalized.to_csv("datasets/phase1/test/y4_test.csv", index=False)

print("Features normalized: ASA, pH, T_(C)")
print("All y values normalized")
