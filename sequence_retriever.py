import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
from io import StringIO
import os
import sys
from Bio import SeqIO

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Import from database file
from uniprot_database import load_local_uniprot_database


def get_sequence_from_uniprot(uniprot_id, max_retries=5):
    """Function to retrieve protein sequences from UniProt API

    Args:
        uniprot_id (str): UniProt ID to retrieve
        max_retries (int): Maximum number of retries for failed requests

    Returns:
        str: Protein sequence or None if not found
    """
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


def load_additional_fasta(fasta_file):
    """Load additional local FASTA database"""
    if not os.path.exists(fasta_file):
        print(f"Warning: Additional database file {fasta_file} not found.")
        return {}

    print(f"Loading additional database from {fasta_file}...")
    uniprot_dict = {}

    if TQDM_AVAILABLE:
        # Count records first
        with open(fasta_file, "r") as f:
            record_count = sum(1 for line in f if line.startswith(">"))

        # Parse with progress bar
        with tqdm(
            total=record_count, desc="Parsing additional database", unit="seq"
        ) as pbar:
            for record in SeqIO.parse(fasta_file, "fasta"):
                uniprot_id = record.id.split("|")[1] if "|" in record.id else record.id
                uniprot_dict[uniprot_id] = str(record.seq)
                pbar.update(1)
    else:
        for record in SeqIO.parse(fasta_file, "fasta"):
            uniprot_id = record.id.split("|")[1] if "|" in record.id else record.id
            uniprot_dict[uniprot_id] = str(record.seq)

    print(f"Loaded {len(uniprot_dict)} sequences from additional database")
    return uniprot_dict


def process_uniprot_ids(
    input_file,
    output_file,
    db_type="sprot",
    taxonomy_id=None,
    use_api=False,
    additional_db=None,
):
    """Process UniProt IDs from input file and retrieve their sequences

    Args:
        input_file (str): Path to input TSV file with UniProt_ID column
        output_file (str): Path to output TSV file
        db_type (str): Database type to use - 'sprot', 'trembl', or 'both'
        taxonomy_id (int, optional): Taxonomy ID to filter database by
        use_api (bool): Whether to look up missing IDs via UniProt API
        additional_db (str, optional): Path to additional local FASTA database

    Returns:
        None: Writes results to output_file
    """
    try:
        data = pd.read_csv(input_file, sep="\t")

        if "UniProt_ID" not in data.columns:
            print("Error: Input file must contain a 'UniProt_ID' column")
            return

        unique_uniprot_ids = data["UniProt_ID"].unique()
        print(f"Found {len(unique_uniprot_ids)} unique UniProt IDs in the input file")

        # Store all found sequences
        found_sequences = {}

        # Step 1: Try to find sequences in the additional database
        if additional_db:
            additional_dict = load_additional_fasta(additional_db)

            # Look up IDs in the additional database
            additional_found = 0
            for uniprot_id in unique_uniprot_ids:
                if uniprot_id in additional_dict:
                    found_sequences[uniprot_id] = additional_dict[uniprot_id]
                    additional_found += 1

            print(f"Found {additional_found} sequences in the additional database")

        # Step 2: For remaining IDs, try the downloaded database
        remaining_ids = [id for id in unique_uniprot_ids if id not in found_sequences]
        if remaining_ids:
            print(
                f"Looking up {len(remaining_ids)} remaining IDs in the downloaded database..."
            )

            # Load downloaded database
            uniprot_dict = load_local_uniprot_database(
                db_type=db_type, taxonomy_id=taxonomy_id
            )

            # Find matches
            download_found = 0
            for uniprot_id in remaining_ids:
                if uniprot_id in uniprot_dict:
                    found_sequences[uniprot_id] = uniprot_dict[uniprot_id]
                    download_found += 1

            print(
                f"Found {download_found} additional sequences in the downloaded database"
            )

        # Step 3: For still remaining IDs, try API if enabled
        remaining_ids = [id for id in unique_uniprot_ids if id not in found_sequences]
        if use_api and remaining_ids:
            print(f"Looking up {len(remaining_ids)} remaining IDs via UniProt API...")

            api_found = 0
            if TQDM_AVAILABLE:
                for uniprot_id in tqdm(remaining_ids, desc="API requests"):
                    seq = get_sequence_from_uniprot(uniprot_id)
                    if seq:
                        found_sequences[uniprot_id] = seq
                        api_found += 1
            else:
                for i, uniprot_id in enumerate(remaining_ids):
                    if i % 10 == 0:
                        print(f"Processing {i}/{len(remaining_ids)} API requests...")
                    seq = get_sequence_from_uniprot(uniprot_id)
                    if seq:
                        found_sequences[uniprot_id] = seq
                        api_found += 1

            print(f"Found {api_found} additional sequences via API")

        # Apply to dataframe and save
        data["Protein_Sequence"] = data["UniProt_ID"].map(found_sequences)
        data.to_csv(output_file, sep="\t", index=False)

        print(f"Processed {len(found_sequences)} unique UniProt IDs")
        print(
            f"Success rate: {len(found_sequences)}/{len(unique_uniprot_ids)} ({len(found_sequences)/len(unique_uniprot_ids)*100:.1f}%)"
        )

    except Exception as e:
        print(f"An error occurred during processing: {str(e)}")


if __name__ == "__main__":
    """Command line interface for processing UniProt IDs"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Process UniProt IDs and retrieve protein sequences"
    )
    parser.add_argument("input_file", help="Input TSV file with UniProt_ID column")
    parser.add_argument("output_file", help="Output TSV file")
    parser.add_argument(
        "--db-type",
        choices=["sprot", "trembl", "both"],
        default="sprot",
        help="Database type for download",
    )
    parser.add_argument(
        "--taxonomy",
        type=int,
        help="Taxonomy ID to filter downloaded database",
    )
    parser.add_argument(
        "--use-api", action="store_true", help="Look up missing IDs via UniProt API"
    )
    parser.add_argument(
        "--additional-db", help="Path to additional local FASTA database"
    )

    args = parser.parse_args()

    process_uniprot_ids(
        args.input_file,
        args.output_file,
        db_type=args.db_type,
        taxonomy_id=args.taxonomy,
        use_api=args.use_api,
        additional_db=args.additional_db,
    )
