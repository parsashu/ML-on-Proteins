import sys
import os
import numpy as np

sys.path.append(os.path.abspath("."))
from utils.improve_tool.pre_processing import pre_processing
from utils.improve_tool.predict import predict_stability


def improve_stability(json_data, use_gpu=True):
    original_seq = json_data["Protein_Sequence"]
    X_norm, seqs_list = pre_processing(json_data, use_gpu=use_gpu)
    predictions = predict_stability(X_norm, use_gpu=use_gpu)

    # Find the index with the highest prediction
    best_index = np.argmax(predictions)
    best_seq = seqs_list[best_index]

    # Find changed amino acid
    is_changed = False
    changed_position = None

    if best_seq != original_seq:
        is_changed = True

        for i, (orig_aa, best_aa) in enumerate(zip(original_seq, best_seq)):
            if orig_aa != best_aa:
                original_aa = orig_aa
                changed_aa = best_aa
                changed_position = i + 1
                break

    # Calculate the change in Tm
    tm_change = None
    if is_changed:
        original_prediction = predictions[0]
        best_prediction = predictions[best_index]
        tm_change = best_prediction - original_prediction
    else:
        tm_change = 0.0

    return {
        "best_sequence": best_seq,
        "changed_position": changed_position,
        "original_amino_acid": original_aa,
        "changed_amino_acid": changed_aa,
        "tm_change": tm_change,
        "is_changed": is_changed,
    }


# json_data = {
#     "ASA": 9.0,
#     "pH": 6.0,
#     "Tm_(C)": 80.4,
#     "Coil": 1,
#     "Helix": 0,
#     "Sheet": 0,
#     "Turn": 0,
#     "Protein_Sequence": "MGDVEKGKKIFVQKCAQCHTVEKGGKHKTGPNLHGLFGRKTGQAPGFTYTDANKNKGITWKEETLMEYLENPKKYIPGTKMIFAGIKKKTEREDLIAYLKKATNE",
# }

# result = improve_stability(json_data)
# print(result)

# output = {
#     "best_sequence": "MGDVEKGKKIFVQKCAQCETVEKGGKHKTGPNLHGLFGRKTGQAPGFTYTDANKNKGITWKEETLMEYLENPKKYIPGTKMIFAGIKKKTEREDLIAYLKKATNE",
#     "changed_position": 19,
#     "original_amino_acid": "H",
#     "changed_amino_acid": "E",
#     "tm_change": np.float64(10.839448094582622),
#     "is_changed": True,
# }
