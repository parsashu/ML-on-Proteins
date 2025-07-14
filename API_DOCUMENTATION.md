# Protein Stability Improvement API

A REST API for improving protein stability through amino acid substitution predictions.

## Base URL

```
https://stable-protein.subnitro.pro
```

## Endpoints

### 1. Health Check

**GET** `/health`

Check if the API is running.

**Response:**

```json
{
  "status": "healthy",
  "message": "Protein Stability API is running"
}
```

### 2. Improve Stability

**POST** `/improve-stability`

Find the best amino acid substitution to improve protein stability.

## Request Parameters

| Parameter        | Type   | Required | Description                                   |
| ---------------- | ------ | -------- | --------------------------------------------- |
| ASA              | float  | Yes      | Accessible Surface Area                       |
| pH               | float  | Yes      | pH value                                      |
| Tm\_(C)          | float  | Yes      | Melting temperature in Celsius                |
| Coil             | int    | Yes      | Binary indicator for coil structure (0 or 1)  |
| Helix            | int    | Yes      | Binary indicator for helix structure (0 or 1) |
| Sheet            | int    | Yes      | Binary indicator for sheet structure (0 or 1) |
| Turn             | int    | Yes      | Binary indicator for turn structure (0 or 1)  |
| Protein_Sequence | string | Yes      | Amino acid sequence                           |

## Response Format

### Success Response

```json
{
  "success": true,
  "data": {
    "best_sequence": "MGDVEKGKKIFVQKCAQCETVEKGGKHKTGPNLHGLFGRKTGQAPGFTYTDANKNKGITWKEETLMEYLENPKKYIPGTKMIFAGIKKKTEREDLIAYLKKATNE",
    "changed_position": 19,
    "original_amino_acid": "H",
    "changed_amino_acid": "E",
    "tm_change": 10.84,
    "is_changed": true
  }
}
```

### Error Response

```json
{
  "success": false,
  "error": "Missing required fields: ASA, pH"
}
```

## Usage Examples

### Python Example

```python
import requests
import json

# API endpoint
url = "http://stable-protein.subnitro.pro/improve-stability"

# Request data
data = {
    "ASA": 9.0,
    "pH": 6.0,
    "Tm_(C)": 80.4,
    "Coil": 1,
    "Helix": 0,
    "Sheet": 0,
    "Turn": 0,
    "Protein_Sequence": "MGDVEKGKKIFVQKCAQCHTVEKGGKHKTGPNLHGLFGRKTGQAPGFTYTDANKNKGITWKEETLMEYLENPKKYIPGTKMIFAGIKKKTEREDLIAYLKKATNE"
}

# Make request
response = requests.post(url, json=data, headers={"Content-Type": "application/json"})

# Parse response
if response.status_code == 200:
    result = response.json()
    if result.get("success"):
        data = result["data"]
        print(f"Best mutation: {data['original_amino_acid']}{data['changed_position']}{data['changed_amino_acid']}")
        print(f"Tm improvement: +{data['tm_change']:.2f}°C")
    else:
        print(f"Error: {result.get('error')}")
else:
    print(f"Request failed with status {response.status_code}")
```

### cURL Example

```bash
curl -X POST http://stable-protein.subnitro.pro/improve-stability \
  -H "Content-Type: application/json" \
  -d '{
    "ASA": 9.0,
    "pH": 6.0,
    "Tm_(C)": 80.4,
    "Coil": 1,
    "Helix": 0,
    "Sheet": 0,
    "Turn": 0,
    "Protein_Sequence": "MGDVEKGKKIFVQKCAQCHTVEKGGKHKTGPNLHGLFGRKTGQAPGFTYTDANKNKGITWKEETLMEYLENPKKYIPGTKMIFAGIKKKTEREDLIAYLKKATNE"
  }'
```

### JavaScript Example

```javascript
const url = "http://stable-protein.subnitro.pro/improve-stability";
const data = {
  ASA: 9.0,
  pH: 6.0,
  "Tm_(C)": 80.4,
  Coil: 1,
  Helix: 0,
  Sheet: 0,
  Turn: 0,
  Protein_Sequence:
    "MGDVEKGKKIFVQKCAQCHTVEKGGKHKTGPNLHGLFGRKTGQAPGFTYTDANKNKGITWKEETLMEYLENPKKYIPGTKMIFAGIKKKTEREDLIAYLKKATNE",
};

fetch(url, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify(data),
})
  .then((response) => response.json())
  .then((result) => {
    if (result.success) {
      const data = result.data;
      console.log(
        `Best mutation: ${data.original_amino_acid}${data.changed_position}${data.changed_amino_acid}`
      );
      console.log(`Tm improvement: +${data.tm_change.toFixed(2)}°C`);
    } else {
      console.error(`Error: ${result.error}`);
    }
  })
  .catch((error) => console.error("Request failed:", error));
```

## Response Fields

| Field               | Type   | Description                                   |
| ------------------- | ------ | --------------------------------------------- |
| best_sequence       | string | The optimized protein sequence                |
| changed_position    | int    | Position of the amino acid change (1-indexed) |
| original_amino_acid | string | Original amino acid at the position           |
| changed_amino_acid  | string | New amino acid at the position                |
| tm_change           | float  | Predicted change in melting temperature (°C)  |
| is_changed          | bool   | Whether a mutation was recommended            |

## Notes

- The API processes 371 different mutations (original sequence + 370 variants)
- Processing time is typically 1-2 minutes for a single protein
- The API uses machine learning models to predict stability improvements
- All amino acid positions are 1-indexed (first position is 1, not 0)

## Error Codes

| Status Code | Description                  |
| ----------- | ---------------------------- |
| 200         | Success                      |
| 400         | Bad Request (missing fields) |
| 405         | Method Not Allowed           |
| 500         | Internal Server Error        |
