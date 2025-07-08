from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import json
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.improve_stability_tool import improve_stability

app = Flask(__name__)
CORS(app)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


app.json_encoder = NumpyEncoder


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "Protein Stability API is running"})


@app.route("/improve-stability", methods=["POST"])
def improve_stability_endpoint():
    """
    Endpoint to improve protein stability

    Expected JSON payload:
    {
        "ASA": 9.0,
        "pH": 6.0,
        "Tm_(C)": 80.4,
        "Coil": 1,
        "Helix": 0,
        "Sheet": 0,
        "Turn": 0,
        "Protein_Sequence": "MGDVEKGKKIFVQKCAQCHTVEKGGKHKTGPNLHGLFGRKTGQAPGFTYTDANKNKGITWKEETLMEYLENPKKYIPGTKMIFAGIKKKTEREDLIAYLKKATNE"
    }
    """
    try:
        # Get JSON data from request
        json_data = request.get_json()

        if not json_data:
            return jsonify({"error": "No JSON data provided"}), 400

        # Validate required fields
        required_fields = [
            "ASA",
            "pH",
            "Tm_(C)",
            "Coil",
            "Helix",
            "Sheet",
            "Turn",
            "Protein_Sequence",
        ]
        missing_fields = [field for field in required_fields if field not in json_data]

        if missing_fields:
            return (
                jsonify(
                    {"error": f"Missing required fields: {', '.join(missing_fields)}"}
                ),
                400,
            )

        # Validate protein sequence
        if not json_data["Protein_Sequence"]:
            return jsonify({"error": "Protein sequence cannot be empty"}), 400

        # Call the improve_stability function
        result = improve_stability(json_data, use_gpu=False)

        return jsonify({"success": True, "data": result})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
