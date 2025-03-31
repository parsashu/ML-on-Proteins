import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from Bio import SeqIO
import os
import gzip
import shutil
import time
import sys

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print(
        "For a better download experience, consider installing tqdm with: pip install tqdm"
    )


def download_uniprot_database(db_type="sprot", force_download=False, taxonomy_id=None):
    """Download the UniProt database

    Args:
        db_type (str): Type of database to download - 'sprot' (smaller, reviewed) or 'trembl' (larger, unreviewed)
        force_download (bool): Force download even if file exists
        taxonomy_id (int): Optional taxonomy ID to filter by (e.g., 9606 for human)

    Returns:
        str: Path to the downloaded and extracted fasta file
    """
    # Set file paths
    if taxonomy_id:
        # With taxonomy filter
        if db_type == "sprot":
            db_url = f"https://rest.uniprot.org/uniprotkb/stream?format=fasta&query=%28taxonomy_id:{taxonomy_id}%29%20AND%20%28reviewed:true%29"
            gz_file = f"uniprot_sprot_taxon{taxonomy_id}.fasta.gz"
            fasta_file = f"uniprot_sprot_taxon{taxonomy_id}.fasta"
        elif db_type == "trembl":
            db_url = f"https://rest.uniprot.org/uniprotkb/stream?format=fasta&query=%28taxonomy_id:{taxonomy_id}%29%20AND%20%28reviewed:false%29"
            gz_file = f"uniprot_trembl_taxon{taxonomy_id}.fasta.gz"
            fasta_file = f"uniprot_trembl_taxon{taxonomy_id}.fasta"
        elif db_type == "both":
            # Both reviewed and unreviewed for a taxonomy
            db_url = f"https://rest.uniprot.org/uniprotkb/stream?format=fasta&query=taxonomy_id:{taxonomy_id}"
            gz_file = f"uniprot_both_taxon{taxonomy_id}.fasta.gz"
            fasta_file = f"uniprot_both_taxon{taxonomy_id}.fasta"
    else:
        # Original code for full databases
        if db_type == "sprot":
            db_url = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz"
            gz_file = "uniprot_sprot.fasta.gz"
            fasta_file = "uniprot_sprot.fasta"
        elif db_type == "trembl":
            db_url = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_trembl.fasta.gz"
            gz_file = "uniprot_trembl.fasta.gz"
            fasta_file = "uniprot_trembl.fasta"
        else:
            raise ValueError("db_type must be either 'sprot', 'trembl', or 'both'")

    # If fasta file already exists and we're not forcing download, return the path
    if os.path.exists(fasta_file) and not force_download:
        print(f"Using existing {fasta_file} file")
        return fasta_file

    # Otherwise download the file
    print(f"Downloading {db_type} database from UniProt (this may take a while)...")
    print(f"URL: {db_url}")

    # Create a session with retry strategy
    session = requests.Session()
    retry_strategy = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)

    # Add headers to avoid potential blocking
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Python/3.13 UniProt-Database-Downloader",
        "Accept": "application/fasta",
    }

    try:
        start_time = time.time()

        # Stream download to avoid loading large file into memory
        with session.get(db_url, stream=True, headers=headers) as response:
            response.raise_for_status()
            file_size = int(response.headers.get("content-length", 0))
            output_file = gz_file if db_url.endswith(".gz") else fasta_file

            # For REST API responses without content length or if file_size is 0
            if file_size == 0:
                file_size = None  # tqdm handles None for unknown size

            # Use tqdm for progress bar if available
            if TQDM_AVAILABLE and file_size:
                # Known file size
                with open(output_file, "wb") as f, tqdm(
                    desc=f"Downloading {os.path.basename(output_file)}",
                    total=file_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    bar_format="{desc}: {percentage:3.1f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            elif TQDM_AVAILABLE:
                # Unknown file size
                with open(output_file, "wb") as f, tqdm(
                    desc=f"Downloading {os.path.basename(output_file)}",
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    bar_format="{desc}: {n_fmt} [{elapsed}, {rate_fmt}]",
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            else:
                # Fallback to simple progress reporting if tqdm is not available
                with open(output_file, "wb") as f:
                    downloaded = 0
                    last_percent = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)

                            # Print progress
                            if file_size:
                                percent = int(downloaded * 100 / file_size)
                                if percent > last_percent:
                                    sys.stdout.write(
                                        f"\rDownloading: {percent}% ({downloaded/(1024*1024):.1f} MB)"
                                    )
                                    sys.stdout.flush()
                                    last_percent = percent
                            else:
                                # Print progress every 10MB for unknown size
                                if downloaded % (10 * 1024 * 1024) < 8192:
                                    mb_downloaded = downloaded / (1024 * 1024)
                                    elapsed = time.time() - start_time
                                    sys.stdout.write(
                                        f"\rDownloaded {mb_downloaded:.1f} MB in {elapsed:.1f} seconds"
                                    )
                                    sys.stdout.flush()

                    # Print newline after progress reporting
                    print()

        elapsed_time = time.time() - start_time
        print(f"Download completed in {elapsed_time:.1f} seconds.")

        # Extract if it's a gzipped file
        if db_url.endswith(".gz"):
            print(f"Extracting {gz_file}...")

            # Use tqdm for extraction progress if available
            if TQDM_AVAILABLE:
                # Get the size of the gzipped file for the progress bar
                gz_size = os.path.getsize(gz_file)

                with gzip.open(gz_file, "rb") as f_in:
                    with open(fasta_file, "wb") as f_out:
                        with tqdm(
                            desc=f"Extracting {os.path.basename(gz_file)}",
                            total=gz_size,
                            unit="B",
                            unit_scale=True,
                            unit_divisor=1024,
                        ) as pbar:
                            # Read and write in chunks to show progress
                            chunk_size = 8192
                            while True:
                                chunk = f_in.read(chunk_size)
                                if not chunk:
                                    break
                                f_out.write(chunk)
                                pbar.update(
                                    chunk_size
                                )  # This isn't exact but provides visual feedback
            else:
                # Simple extraction without progress bar
                with gzip.open(gz_file, "rb") as f_in:
                    with open(fasta_file, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)

            # Remove the gz file to save space
            os.remove(gz_file)
            print(f"Extraction complete.")

        # Verify file size
        file_stats = os.stat(fasta_file)
        file_size_mb = file_stats.st_size / (1024 * 1024)
        print(f"Final file size: {file_size_mb:.1f} MB")

        if file_size_mb < 100 and "taxon" not in fasta_file and db_type != "both":
            print(
                "WARNING: Downloaded file is smaller than expected. There might be an issue with the download."
            )
            print(
                "You may want to try a different method or check the UniProt website for updates."
            )

        return fasta_file

    except Exception as e:
        print(f"Error downloading or extracting UniProt database: {str(e)}")
        return None


def load_local_uniprot_database(fasta_file=None, db_type="sprot", taxonomy_id=None):
    """Function to load UniProt database from local FASTA file

    Args:
        fasta_file (str, optional): Path to the FASTA file. If None, it will be downloaded.
        db_type (str, optional): Type of database - 'sprot', 'trembl', or 'both'.
        taxonomy_id (int, optional): Taxonomy ID to filter by.

    Returns:
        dict: Dictionary mapping UniProt IDs to sequences
    """
    # If no file is specified, download or use existing one
    if fasta_file is None:
        fasta_file = download_uniprot_database(db_type=db_type, taxonomy_id=taxonomy_id)
        if fasta_file is None:
            return {}

    print(f"Loading local UniProt database from {fasta_file}...")
    uniprot_dict = {}

    if TQDM_AVAILABLE:
        # Count the total number of records first (faster than calculating file size)
        with open(fasta_file, "r") as f:
            record_count = sum(1 for line in f if line.startswith(">"))

        # Parse with progress bar
        with tqdm(total=record_count, desc="Parsing sequences", unit="seq") as pbar:
            for record in SeqIO.parse(fasta_file, "fasta"):
                uniprot_id = record.id.split("|")[1] if "|" in record.id else record.id
                uniprot_dict[uniprot_id] = str(record.seq)
                pbar.update(1)
    else:
        # Parse without progress bar
        for record in SeqIO.parse(fasta_file, "fasta"):
            uniprot_id = record.id.split("|")[1] if "|" in record.id else record.id
            uniprot_dict[uniprot_id] = str(record.seq)

    print(f"Loaded {len(uniprot_dict)} sequences from local database")
    return uniprot_dict


if __name__ == "__main__":
    """Command line interface for downloading UniProt databases"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download and process UniProt database files"
    )
    parser.add_argument(
        "--db-type",
        choices=["sprot", "trembl", "both"],
        default="sprot",
        help="Database type: sprot (smaller, reviewed), trembl (larger, unreviewed), or both",
    )
    parser.add_argument(
        "--taxonomy",
        type=int,
        help="Optional taxonomy ID to filter by (e.g., 9606 for human)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Force download even if the file exists"
    )
    parser.add_argument("--output", help="Output file path (optional)")

    args = parser.parse_args()

    # Download the database
    output_file = args.output if args.output else None
    result = download_uniprot_database(
        db_type=args.db_type, force_download=args.force, taxonomy_id=args.taxonomy
    )

    if result:
        print(f"Database downloaded successfully to: {result}")
    else:
        print("Failed to download the database")
