import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv("datasets/protein_dataset.tsv", sep="\t")

df["SEC_STR_ENCODED"] = df["SEC_STR_ENCODED"].apply(
    lambda x: eval(x) if isinstance(x, str) else x
)
df["Coil"] = df["SEC_STR_ENCODED"].apply(lambda x: x[0])
df["Helix"] = df["SEC_STR_ENCODED"].apply(lambda x: x[1])
df["Sheet"] = df["SEC_STR_ENCODED"].apply(lambda x: x[2])
df["Turn"] = df["SEC_STR_ENCODED"].apply(lambda x: x[3])

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

features_df = df[features].copy()
features_df = features_df.fillna(-1000)


train_df, test_df = train_test_split(features_df, test_size=0.1, random_state=42)

train_df.to_csv("datasets/phase2/train/X_train.csv", index=False)
test_df.to_csv("datasets/phase2/test/X_test.csv", index=False)

print("Features split and saved to train_features.csv and test_features.csv")
