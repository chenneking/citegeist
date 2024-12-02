import hashlib
import json
import pandas as pd
from pymilvus import connections, Collection

# Establish Milvus connection
connections.connect("dataset_mini.db")
client = connections.get_connection("default")

# Function to compute SHA256 hash
def compute_hash(entry: dict) -> str:
    """Compute a SHA256 hash for a given entry."""
    entry_str = json.dumps(entry, sort_keys=True)
    return hashlib.sha256(entry_str.encode('utf-8')).hexdigest()

# Function to compute embedding (placeholder example)
def compute_embedding(text: str) -> list:
    """Compute embedding for a given text (mockup)."""
    return [len(text)] * 128  # Replace with actual embedding computation logic

# Function to load existing database into a hash table
def load_database_into_hash_table(collection_name: str) -> dict:
    """Load database into a hash table with precomputed hashes."""
    collection = Collection(collection_name)
    hash_table = {}
    for entry in collection.query(expr=None, output_fields=["id", "hash", "abstract"]):
        hash_table[entry["hash"]] = entry
    return hash_table

# Reload and process Kaggle data
def reload_and_process_data(kaggle_file: str, collection_name: str, hash_table: dict):
    """Reload data from Kaggle and process updates to the database."""
    new_data = pd.read_csv(kaggle_file)  # Assuming CSV format for Kaggle data

    for _, row in new_data.iterrows():
        metadata = row.to_dict()
        metadata_hash = compute_hash(metadata)

        # Case 1: Entry with hash already exists
        if metadata_hash in hash_table:
            continue  # Skip existing entries

        # Case 2: Hash doesn't exist but ID is found
        existing_entries = client.query(
            collection_name=collection_name,
            expr=f'id == "{metadata["id"]}"',
            output_fields=["id", "hash", "abstract"]
        )
        if existing_entries:
            existing_entry = existing_entries[0]
            if existing_entry["abstract"] != metadata["abstract"]:
                # Update entry: replace abstract and recompute hash and embedding
                updated_entry = metadata.copy()
                updated_entry["hash"] = metadata_hash
                updated_entry["embedding"] = compute_embedding(metadata["abstract"])
                client.update(collection_name=collection_name, data=[updated_entry])
                print(f"Updated entry for ID {metadata['id']}.")
            continue  # Skip further processing for this entry

        # Case 3: ID not found; insert new row
        new_entry = metadata.copy()
        new_entry["hash"] = metadata_hash
        new_entry["embedding"] = compute_embedding(metadata["abstract"])
        client.insert(collection_name=collection_name, data=[new_entry])
        print(f"Inserted new entry for ID {metadata['id']}.")

# Main execution
if __name__ == "__main__":
    # Configuration
    kaggle_file = "/path/to/kaggle/dataset.csv"
    collection_name = "arxiv_metadata"

    # Load existing database into a hash table
    print("Loading database into hash table...")
    hash_table = load_database_into_hash_table(collection_name)
    print(f"Hash table loaded with {len(hash_table)} entries.")

    # Reload and process Kaggle data
    print("Processing Kaggle data...")
    reload_and_process_data(kaggle_file, collection_name, hash_table)
    print("Processing complete.")
