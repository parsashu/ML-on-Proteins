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
   git clone https://github.com/parsashu/ML-in-physics.git
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
