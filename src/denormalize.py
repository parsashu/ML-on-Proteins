import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

feature_scaler_path = "../models/scalers/feature_scaler.joblib"
target_scaler_path = "../models/scalers/target_scaler.joblib"


def load_scalers():
    scaler = joblib.load(feature_scaler_path)
    y_scaler = joblib.load(target_scaler_path)
    return scaler, y_scaler


def denormalize_features(normalized_features):
    scaler, _ = load_scalers()
    return scaler.inverse_transform(normalized_features)


def denormalize_predictions(normalized_predictions):
    _, y_scaler = load_scalers()
    if len(normalized_predictions.shape) == 1:
        normalized_predictions = normalized_predictions.reshape(-1, 1)
    return y_scaler.inverse_transform(normalized_predictions)
