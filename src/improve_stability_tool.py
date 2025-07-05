import sys
import os
import numpy as np

sys.path.append(os.path.abspath("."))
from utils.improve_tool.pre_processing import pre_processing
from utils.improve_tool.predict import predict_stability


def improve_stability(json_data):
    original_seq = json_data["Protein_Sequence"]
    X_norm, seqs_list = pre_processing(json_data)
    predictions = predict_stability(X_norm)

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

    return best_seq, changed_position, original_aa, changed_aa, is_changed


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

best_seq, changed_position, original_aa, changed_aa, is_changed = improve_stability(json_data)
print(f"Original protein sequence: {json_data['Protein_Sequence']}")
print(f"Best protein sequence: {best_seq}")
print(f"Changed position: {changed_position}")
print(f"Original amino acid: {original_aa}")
print(f"New amino acid: {changed_aa}")
print(f"Is changed: {is_changed}")
