from Bio import SeqIO
from Bio import Entrez
from io import StringIO
import requests

Entrez.email = "your.email@example.com"  

def get_sequence_from_uniprot(uniprot_id):
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    response = requests.get(url)
    if response.status_code == 200:
        fasta = StringIO(response.text)
        for record in SeqIO.parse(fasta, "fasta"):
            return record.seq
    return None

seq = get_sequence_from_uniprot("P01005")
print(seq)