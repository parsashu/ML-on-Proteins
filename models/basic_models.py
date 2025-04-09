import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
# Load normalized datasets
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

# Initialize models
models = {
    # "Decision Tree": [
    #     ("Tm", DecisionTreeRegressor(random_state=42)),
    #     ("m", DecisionTreeRegressor(random_state=42)),
    #     ("Cm", DecisionTreeRegressor(random_state=42)),
    #     ("deltaG", DecisionTreeRegressor(random_state=42)),
    # ],
    "Random Forest": [
        ("Tm", RandomForestRegressor(n_estimators=100, random_state=42)),
        ("m", RandomForestRegressor(n_estimators=100, random_state=42)),
        ("Cm", RandomForestRegressor(n_estimators=100, random_state=42)),
        ("deltaG", RandomForestRegressor(n_estimators=100, random_state=42)),
    ],
    # "SVR": [
    #     ("Tm", SVR(kernel="rbf", C=1.0, epsilon=0.1)),
    #     ("m", SVR(kernel="rbf", C=1.0, epsilon=0.1)),
    #     ("Cm", SVR(kernel="rbf", C=1.0, epsilon=0.1)),
    #     ("deltaG", SVR(kernel="rbf", C=1.0, epsilon=0.1)),
    # ],
    # "Gradient Boosting": [
    #     ("Tm", GradientBoostingRegressor(n_estimators=100, random_state=42)),
    #     ("m", GradientBoostingRegressor(n_estimators=100, random_state=42)),
    #     ("Cm", GradientBoostingRegressor(n_estimators=100, random_state=42)),
    #     ("deltaG", GradientBoostingRegressor(n_estimators=100, random_state=42)),
    # ],
    # "Neural Network": [
    #     ("Tm", MLPRegressor(hidden_layer_sizes=(200, 100), max_iter=1000)),
    #     ("m", MLPRegressor(hidden_layer_sizes=(200, 100), max_iter=1000)),
    #     ("Cm", MLPRegressor(hidden_layer_sizes=(200, 100), max_iter=1000)),
    #     ("deltaG", MLPRegressor(hidden_layer_sizes=(200, 100), max_iter=1000)),
    # ],
}

# Training data pairs
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


# Calculate metrics
def print_metrics(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{model_name} Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")


# Train models with progress bar
for model_type, model_list in models.items():
    print(f"\nTraining {model_type} models:")
    for (target, model), (X_train, y_train), (X_test, y_test) in tqdm(
        zip(model_list, train_data, test_data),
        total=len(model_list),
        desc=f"{model_type}",
    ):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"\n{model_type} - {target} Metrics:")
        print_metrics(y_test, y_pred, f"{model_type} - {target}")


# # Feature importance analysis
# def print_feature_importance(model, feature_names, model_name):
#     importances = model.feature_importances_
#     indices = np.argsort(importances)[::-1]

#     print(f"\n{model_name} Top 10 Important Features:")
#     for f in range(min(10, len(feature_names))):
#         print(f"{feature_names[indices[f]]}: {importances[indices[f]]:.4f}")


# # Print feature importance for each model
# feature_names = X1_train.columns.tolist()
# print_feature_importance(dt1, feature_names, "Model 1 (Tm)")
# print_feature_importance(dt2, feature_names, "Model 2 (m)")
# print_feature_importance(dt3, feature_names, "Model 3 (Cm)")
# print_feature_importance(dt4, feature_names, "Model 4 (deltaG)")

# # Print feature importance for Random Forest models
# print("\n=== Random Forest Feature Importance ===")
# print_feature_importance(rf1, feature_names, "Random Forest Model 1 (Tm)")
# print_feature_importance(rf2, feature_names, "Random Forest Model 2 (m)")
# print_feature_importance(rf3, feature_names, "Random Forest Model 3 (Cm)")
# print_feature_importance(rf4, feature_names, "Random Forest Model 4 (deltaG)")
