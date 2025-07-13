import requests
import json

# API Configuration
url = "http://localhost:5000/improve-stability"
headers = {"Content-Type": "application/json"}

# Protein data payload
payload = {
    "ASA": 9.0,
    "pH": 6.0,
    "Tm_(C)": 80.4,
    "Coil": 1,
    "Helix": 0,
    "Sheet": 0,
    "Turn": 0,
    "Protein_Sequence": "MGDVEKGKKIFVQKCAQCHTVEKGGKHKTGPNLHGLFGRKTGQAPGFTYTDANKNKGITWKEETLMEYLENPKKYIPGTKMIFAGIKKKTEREDLIAYLKKATNE",
}

# Make API request
response = requests.post(url, headers=headers, json=payload)

# Print response
print("Status Code:", response.status_code)
print("Response:")
print(json.dumps(response.json(), indent=2))

# Extract result
if response.status_code == 200:
    result = response.json()
    if result.get("success"):
        data = result["data"]
        print(
            f"\nBest mutation: {data['original_amino_acid']}{data['changed_position']}{data['changed_amino_acid']}"
        )
        print(f"Tm improvement: +{data['tm_change']:.2f}°C")
    else:
        print("Error:", result.get("error"))
else:
    print("Request failed")
