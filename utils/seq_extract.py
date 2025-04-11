"""
Protein Sequence Extraction Utility

This script retrieves protein sequences from UniProt IDs using both local database
and UniProt REST API. It processes a TSV file containing UniProt IDs and produces
an output file with the corresponding protein sequences.
"""
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from Bio import SeqIO
from io import StringIO
import pandas as pd

raw_file = "datasets/raw/protherm.tsv"
dataset = "datasets/raw/protein_seq.tsv"

def get_sequence_from_uniprot(uniprot_id, max_retries=5):
    """Function to retrieve protein sequences from UniProt API"""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"

    session = requests.Session()
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Python/3.13 Protein-Sequence-Retrieval"
    }

    try:
        response = session.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            fasta = StringIO(response.text)
            for record in SeqIO.parse(fasta, "fasta"):
                return str(record.seq)
            
    except Exception as e:
        print(f"Error retrieving sequence for {uniprot_id}: {str(e)}")

    return None


def load_local_uniprot_database(fasta_file="datasets/uniprot_sprot.fasta"):
    """Function to load UniProt database from local FASTA file"""
    print("Loading local UniProt database...")
    uniprot_dict = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        uniprot_id = record.id.split("|")[1] if "|" in record.id else record.id
        uniprot_dict[uniprot_id] = str(record.seq)
    print(f"Loaded {len(uniprot_dict)} sequences from local database")
    return uniprot_dict


def process_uniprot_ids(input_file, output_file, use_api=True):
    """Process UniProt IDs from input file, retrieve sequences"""
    try:
        data = pd.read_csv(input_file, sep="\t")
        unique_uniprot_ids = data["UniProt_ID"].unique()
        
        # Local database
        uniprot_dict = load_local_uniprot_database()
        found_sequences = {}
        
        for uniprot_id in unique_uniprot_ids:
            if uniprot_id in uniprot_dict:
                found_sequences[uniprot_id] = uniprot_dict[uniprot_id]
        
        # API
        api_found = 0
        if use_api:
            remaining_ids = [id for id in unique_uniprot_ids if id not in found_sequences]
            
            for uniprot_id in remaining_ids:
                seq = get_sequence_from_uniprot(uniprot_id)
                if seq:
                    found_sequences[uniprot_id] = seq
                    api_found += 1
            
            print(f"Found {api_found} additional sequences via API")
        
        data["Protein_Sequence"] = data["UniProt_ID"].map(found_sequences)
        data.to_csv(output_file, sep="\t", index=False)

        print(f"Processed {len(found_sequences)} unique UniProt IDs")
        print(
            f"Success rate: {len(found_sequences)}/{len(unique_uniprot_ids)} ({len(found_sequences)/len(unique_uniprot_ids)*100:.1f}%)"
        )

    except Exception as e:
        print(f"An error occurred during processing: {str(e)}")

process_uniprot_ids(raw_file, dataset, use_api=True)