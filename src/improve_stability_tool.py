import sys
import os

sys.path.append(os.path.abspath("."))
from utils.improve_tool.pre_processing import pre_processing
from utils.improve_tool.predict import predict_stability


def improve_stability(json_data):
    X_norm = pre_processing(json_data)
    print(X_norm)
    predictions = predict_stability(X_norm)
    return predictions


json_data = {
    "ASA": 9.0,
    "pH": 6.0,
    "Tm_(C)": 80.4,
    "Coil": 1,
    "Helix": 0,
    "Sheet": 0,
    "Turn": 0,
    "Protein_Sequence": "MGDVEKGKKIFVQKCAQCHTVEKGGKHKTGPNLHGLFGRKTGQAPGFTYTDANKNKGITWKEETLMEYLENPKKYIPGTKMIFAGIKKKTEREDLIAYLKKATNE",
}

predictions = improve_stability(json_data)
print(predictions)
