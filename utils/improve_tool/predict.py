from catboost import CatBoostRegressor
import numpy as np


def predict_stability(X):
    model = CatBoostRegressor()
    model.load_model("models/catboost_final_model.cbm")
    predictions = model.predict(X)
    return predictions
