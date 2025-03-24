from Bio import SeqIO
from io import StringIO
import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def get_sequence_from_uniprot(uniprot_id, max_retries=5):
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
        else:
            print(
                f"Failed to retrieve sequence for {uniprot_id}, status code: {response.status_code}"
            )
    except Exception as e:
        print(f"Error retrieving sequence for {uniprot_id}: {str(e)}")

    return None


def load_local_uniprot_database(fasta_file="datasets/uniprot_sprot.fasta"):
    print("Loading local UniProt database...")
    uniprot_dict = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        uniprot_id = record.id.split("|")[1] if "|" in record.id else record.id
        uniprot_dict[uniprot_id] = str(record.seq)
    print(f"Loaded {len(uniprot_dict)} sequences from local database")
    return uniprot_dict


def process_uniprot_ids():
    try:
        data = pd.read_csv("datasets/1temp.tsv", sep="\t")

        unique_uniprot_ids = data["UniProt_ID"].unique()

        uniprot_dict = load_local_uniprot_database()

        found_sequences = {}
        for uniprot_id in unique_uniprot_ids:
            if uniprot_id in uniprot_dict:
                found_sequences[uniprot_id] = uniprot_dict[uniprot_id]
            else:
                print(f"Sequence for {uniprot_id} not found in local database")

        data["Protein_Sequence"] = data["UniProt_ID"].map(found_sequences)

        data.to_csv("datasets/1temp_with_sequences.tsv", sep="\t", index=False)

        print(f"Processed {len(found_sequences)} unique UniProt IDs")
        print(
            f"Success rate: {len(found_sequences)}/{len(unique_uniprot_ids)} ({len(found_sequences)/len(unique_uniprot_ids)*100:.1f}%)"
        )

    except Exception as e:
        print(f"An error occurred during processing: {str(e)}")


if __name__ == "__main__":
    process_uniprot_ids()
