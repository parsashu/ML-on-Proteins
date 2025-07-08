from catboost import CatBoostRegressor
import numpy as np


def predict_stability(X, use_gpu=True):
    model = CatBoostRegressor()
    model.load_model("models/catboost_final_model.cbm")
    if use_gpu:
        model.set_params(task_type="GPU")
        model.set_params(devices="0:1")

    predictions = model.predict(X)
    return predictions
