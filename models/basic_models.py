import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
from sklearn.svm import SVR
from sklearnex import patch_sklearn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Enable GPU acceleration for scikit-learn
patch_sklearn()


# Model 1: Tm prediction
X1_train = pd.read_csv("datasets/phase1/train/X1_train.csv")
X1_test = pd.read_csv("datasets/phase1/test/X1_test.csv")
y1_train = pd.read_csv("datasets/phase1/train/y1_train.csv")
y1_test = pd.read_csv("datasets/phase1/test/y1_test.csv")

# Model 2: m prediction
X2_train = pd.read_csv("datasets/phase1/train/X2_train.csv")
X2_test = pd.read_csv("datasets/phase1/test/X2_test.csv")
y2_train = pd.read_csv("datasets/phase1/train/y2_train.csv")
y2_test = pd.read_csv("datasets/phase1/test/y2_test.csv")

# Model 3: Cm prediction
X3_train = pd.read_csv("datasets/phase1/train/X3_train.csv")
X3_test = pd.read_csv("datasets/phase1/test/X3_test.csv")
y3_train = pd.read_csv("datasets/phase1/train/y3_train.csv")
y3_test = pd.read_csv("datasets/phase1/test/y3_test.csv")

# Model 4: deltaG prediction
X4_train = pd.read_csv("datasets/phase1/train/X4_train.csv")
X4_test = pd.read_csv("datasets/phase1/test/X4_test.csv")
y4_train = pd.read_csv("datasets/phase1/train/y4_train.csv")
y4_test = pd.read_csv("datasets/phase1/test/y4_test.csv")

models = {
    # "Decision Tree": [
    #     ("Tm", DecisionTreeRegressor(random_state=42)),
    #     ("m", DecisionTreeRegressor(random_state=42)),
    #     ("Cm", DecisionTreeRegressor(random_state=42)),
    #     ("deltaG", DecisionTreeRegressor(random_state=42)),
    # ],
    "Random Forest": [
        ("Tm", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, warm_start=True)),
        ("m", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, warm_start=True)),
        ("Cm", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, warm_start=True)),
        ("deltaG", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, warm_start=True)),
    ],
    # "Gradient Boosting": [
    #     ("Tm", GradientBoostingRegressor(n_estimators=100, random_state=42)),
    #     ("m", GradientBoostingRegressor(n_estimators=100, random_state=42)),
    #     ("Cm", GradientBoostingRegressor(n_estimators=100, random_state=42)),
    #     ("deltaG", GradientBoostingRegressor(n_estimators=100, random_state=42)),
    # ],
    # "SVR": [
    #     ("Tm", SVR(kernel="rbf", C=1.0, epsilon=0.1)),
    #     ("m", SVR(kernel="rbf", C=1.0, epsilon=0.1)),
    #     ("Cm", SVR(kernel="rbf", C=1.0, epsilon=0.1)),
    #     ("deltaG", SVR(kernel="rbf", C=1.0, epsilon=0.1)),
    # ],
    # "Neural Network": [
    #     ("Tm", nn.Linear(X1_train.shape[1], 1)),
    #     ("m", nn.Linear(X2_train.shape[1], 1)),
    #     ("Cm", nn.Linear(X3_train.shape[1], 1)),
    #     ("deltaG", nn.Linear(X4_train.shape[1], 1)),
    # ],
}

train_data = [
    (X1_train, y1_train.values.ravel()),
    (X2_train, y2_train.values.ravel()),
    (X3_train, y3_train.values.ravel()),
    (X4_train, y4_train.values.ravel()),
]

test_data = [
    (X1_test, y1_test.values.ravel()),
    (X2_test, y2_test.values.ravel()),
    (X3_test, y3_test.values.ravel()),
    (X4_test, y4_test.values.ravel()),
]


def print_metrics(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{model_name} Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")


# Train and evaluate models
for model_type, model_list in models.items():
    print(f"\nTraining {model_type} models:")
    for (target, model), (X_train, y_train), (X_test, y_test) in tqdm(
        zip(model_list, train_data, test_data),
        total=len(model_list),
        desc=f"{model_type}",
    ):
        # Initialize lists to store metrics for each epoch
        mse_history = []
        rmse_history = []
        r2_history = []
        
        # Training loop with 20 epochs
        for epoch in tqdm(range(20), desc=f"Training {target}", leave=False):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # Store metrics
            mse_history.append(mse)
            rmse_history.append(rmse)
            r2_history.append(r2)
            
            # Print progress every 5 epochs
            if (epoch + 1) % 5 == 0:
                print(f"\nEpoch {epoch + 1}/20 - {model_type} - {target}")
                print(f"MSE: {mse:.4f} (Best: {min(mse_history):.4f})")
                print(f"RMSE: {rmse:.4f} (Best: {min(rmse_history):.4f})")
                print(f"R2 Score: {r2:.4f} (Best: {max(r2_history):.4f})")
        
        # Print final metrics
        print(f"\nFinal Metrics for {model_type} - {target}:")
        print(f"Best MSE: {min(mse_history):.4f}")
        print(f"Best RMSE: {min(rmse_history):.4f}")
        print(f"Best R2 Score: {max(r2_history):.4f}")

# def print_feature_importance(model, feature_names, model_name):
#     importances = model.feature_importances_
#     indices = np.argsort(importances)[::-1]

#     print(f"\n{model_name} Top 10 Important Features:")
#     for f in range(min(10, len(feature_names))):
#         print(f"{feature_names[indices[f]]}: {importances[indices[f]]:.4f}")


# feature_names = X1_train.columns.tolist()
# print_feature_importance(dt1, feature_names, "Model 1 (Tm)")
# print_feature_importance(dt2, feature_names, "Model 2 (m)")
# print_feature_importance(dt3, feature_names, "Model 3 (Cm)")
# print_feature_importance(dt4, feature_names, "Model 4 (deltaG)")

# print("\n=== Random Forest Feature Importance ===")
# print_feature_importance(rf1, feature_names, "Random Forest Model 1 (Tm)")
# print_feature_importance(rf2, feature_names, "Random Forest Model 2 (m)")
# print_feature_importance(rf3, feature_names, "Random Forest Model 3 (Cm)")
# print_feature_importance(rf4, feature_names, "Random Forest Model 4 (deltaG)")
