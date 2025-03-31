# Main script that uses the two modules and multiple databases

from uniprot_database import download_uniprot_database, load_local_uniprot_database
from sequence_retriever import process_uniprot_ids, get_sequence_from_uniprot

# Example usage - choose one of these options:

# OPTION 1: Download the complete Swiss-Prot database (~500MB)
# process_uniprot_ids("datasets/8temp.tsv", "datasets/seq_temp.tsv", db_type="sprot")

# OPTION 2: Download all vertebrate proteins (~2GB)
# process_uniprot_ids(
#     "datasets/8temp.tsv", "datasets/seq_temp.tsv", db_type="both", taxonomy_id=7742
# )

# OPTION 3: If you need a database closer to exactly 2GB, try one of these:
process_uniprot_ids(
    input_file="datasets/8temp.tsv",
    output_file="datasets/seq_temp.tsv",
    db_type="both",
    taxonomy_id=40674,  # Mammals
    use_api=True,
    additional_db="datasets/uniprot_sprot.fasta",
)
# process_uniprot_ids("datasets/8temp.tsv", "datasets/seq_temp.tsv", db_type="both", taxonomy_id=33208)  # Metazoa (Animals)

# OPTION 4: To download only proteins from specific organisms:
# process_uniprot_ids("datasets/8temp.tsv", "datasets/seq_temp.tsv", db_type="both", taxonomy_id=9606)   # Human (~800MB)
# process_uniprot_ids("datasets/8temp.tsv", "datasets/seq_temp.tsv", db_type="both", taxonomy_id=10090)  # Mouse
# process_uniprot_ids("datasets/8temp.tsv", "datasets/seq_temp.tsv", db_type="both", taxonomy_id=9031)   # Chicken
