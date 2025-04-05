import requests
import os
from tqdm import tqdm
import gzip
import shutil


def download_uniprot():
    url = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz"

    if not os.path.exists("datasets"):
        os.makedirs("datasets")

    local_file = "datasets/uniprot_sprot.fasta.gz"

    print(f"Downloading UniProt Swiss-Prot database from {url}")

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    with open(local_file, "wb") as f, tqdm(
        desc="Downloading",
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

    print(f"\nDownload complete! File saved to {local_file}")
    print("Extracting gzipped file...")


    with gzip.open(local_file, "rb") as f_in:
        with open(local_file[:-3], "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    os.remove(local_file)
    print(f"Extraction complete! FASTA file available at: {local_file[:-3]}")


if __name__ == "__main__":
    download_uniprot()
