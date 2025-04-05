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
   git clone https://github.com/yourusername/protein-thermodynamics.git
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

## Data Collection
Data for this project is collected from the ProThermDB database, which specializes in thermodynamic properties of proteins. The database allows selection of specific features through its browse -> search functionality.

For this analysis, we selected UniProt protein information to obtain amino acid sequences. We also included PDB, ASA, and Secondary Structure features as they appear relevant to our problem. All thermodynamic parameters were initially included, though we expect to reduce these features later in the process.

The Bio library was used to retrieve amino acid sequences from UniProt IDs.

Source: https://web.iitm.ac.in/bioinfo2/prothermdb/index.html