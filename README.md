## Project Description

This project consists of two distinct phases:

### Phase 1: Predicting Protein Thermodynamic Properties Based on Amino Acid Sequence

This phase aims to develop a computational model to predict thermodynamic properties of proteins based on their amino acid sequences.

### Phase 2: Designing Protein Sequences with Desired Thermodynamic Properties

In this phase, we use generative models (GANs) to produce new sequences with specific and optimized thermodynamic properties. The goal is to create proteins with desired thermal stability and appropriate functional characteristics.

## Installation

To set up the project environment, follow these steps:

1. Clone the repository:

   ```
   git clone https://github.com/parsashu/ML-on-Proteins.git
   cd protein-thermodynamics
   ```

2. Create a virtual environment (optional but recommended):

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:

   ```
   pip install -r requirements.txt
   ```

4. (Optional) If you want to download the UniProt database, run:

   ```
   python utils/download_uniprot.py
   ```

5. (Optional) If you want to extract protein sequences for raw_data.tsv, run:
   ```
   python utils/seq_extract.py
   ```
   This will create protein_seq.tsv with the extracted sequences from UniProt IDs.

## Folder Structure

```
├── datasets/
│   ├── basic_statictical/
│   ├── phase1/
│   │   ├── train/
│   │   └── test/
│   ├── phase2/
│   │   ├── train/
│   │   └── test/
│   ├── pre_processing.ipynb
│   ├── embedding.ipynb
│   ├── raw_data.tsv
│   └── protein_dataset.tsv
├── utils/
│   ├── download_uniprot.py
│   └── seq_extract.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Data Collection

The dataset for this project is obtained from the ProThermDB database. This database allows users to select features for display and perform searches.
🔗 [Database](https://web.iitm.ac.in/bioinfo2/prothermdb/search.html) Link (Data can be explored via the "Browse -> Search" section.)
The downloaded data from this database is stored in datasets/raw_data.tsv. This dataset does not include protein sequences, but the sequences can be retrieved using UniProt_ID.

## API Usage

### Protein Stability Improvement API

A REST API is available for improving protein stability through amino acid substitution predictions.

#### Installation & Setup

1. Install required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the API server:

   ```bash
   python utils/improve_tool/api.py
   ```

   The API will be available at `http://localhost:5000`

#### API Endpoints

##### 1. Health Check

**GET** `/health`

Check if the API is running.

```bash
curl http://localhost:5000/health
```

##### 2. Improve Stability

**POST** `/improve-stability`

Find the best amino acid substitution to improve protein stability.

```bash
curl -X POST http://localhost:5000/improve-stability \
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

##### 3. Predict All Mutations

**POST** `/predict-stability`

Get stability predictions for all possible amino acid substitutions.

#### Python Usage Example

```python
import requests

# API endpoint
url = "http://localhost:5000/improve-stability"

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
response = requests.post(url, json=data)

# Parse response
if response.status_code == 200:
    result = response.json()
    print(f"Best sequence: {result['data']['best_sequence']}")
    print(f"Mutation: {result['data']['original_amino_acid']}{result['data']['changed_position']}{result['data']['changed_amino_acid']}")
    print(f"Tm change: {result['data']['tm_change']:.2f}°C")
else:
    print(f"Error: {response.json()['error']}")
```

#### Request Parameters

| Parameter        | Type   | Description                                   |
| ---------------- | ------ | --------------------------------------------- |
| ASA              | float  | Accessible Surface Area                       |
| pH               | float  | pH value                                      |
| Tm\_(C)          | float  | Melting temperature in Celsius                |
| Coil             | int    | Binary indicator for coil structure (0 or 1)  |
| Helix            | int    | Binary indicator for helix structure (0 or 1) |
| Sheet            | int    | Binary indicator for sheet structure (0 or 1) |
| Turn             | int    | Binary indicator for turn structure (0 or 1)  |
| Protein_Sequence | string | Amino acid sequence                           |

For detailed API documentation, see `utils/improve_tool/api_usage.md`.
