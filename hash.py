import hashlib
import pandas as pd
from pymilvus import MilvusClient
import json

import os
import pandas as pd



# Step 2: Download dataset using Kaggle API
def download_kaggle_dataset(dataset, file_name):
    os.system(f"kaggle datasets download -d {dataset} --unzip")
    print(f"Downloaded {file_name} from Kaggle.")

# Step 3: Load dataset into pandas DataFrame
def load_arxiv_data(file_name):
    print("Loading ArXiv dataset...")
    return pd.read_json(file_name, lines=True)





# Add hash to existing database entries
def compute_hash(entry: dict) -> str:
    print(entry)
    """Compute a SHA256 hash for a given entry."""
    entry_str = json.dumps(entry, sort_keys=True)
    return hashlib.sha256(entry_str.encode('utf-8')).hexdigest()

# Assuming the database has been loaded into a collection
# def add_hash_to_database(collection_name: str):
#     """Add hash attribute to each database entry."""
#     entries = client.query(collection_name=collection_name, expr=None)
#     for entry in entries:
#         print(entry)
#         entry["hash"] = compute_hash(entry)
#         # Update entry in the database with the hash (or save it externally)
#         client.update(collection_name=collection_name, data=[entry])

def add_hash_to_database(collection_name: str, batch_size: int = 100):
    """Add hash attribute to each database entry."""
    offset = 0
    while True:
        # Fetch a batch of entries using limit and offset for pagination
        entries = client.query(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
        )
        if not entries:  # Break if no more entries
            break

        for entry in entries:
            # Compute and add hash
            entry["hash"] = compute_hash(entry)
            # Update the database with the new hash field
            client.update(collection_name=collection_name, data=[entry])

        offset += batch_size  # Move to the next batch





# Reload database from Kaggle
def reload_and_lookup(kaggle_file: str, collection_name: str):
    """Reload data from Kaggle, compute hash, and lookup in database."""
    new_data = pd.read_csv(kaggle_file)  # Replace with correct Kaggle file loading
    for _, row in new_data.iterrows():
        metadata = row.to_dict()
        metadata_hash = compute_hash(metadata)
        # Search for the hash in the database
        matches = client.query(
            collection_name=collection_name,
            expr=f'hash == "{metadata_hash}"',
        )
        if matches:
            print(f"Match found for metadata: {metadata}")
        else:
            print(f"No match found for metadata: {metadata}")


# Step 1: Define dataset parameters
kaggle_dataset = "Cornell-University/arxiv-metadata"  # Replace with your dataset name
dataset_file = "arxiv-metadata-oai-snapshot.json"  # Replace with your dataset's filename


# Run the functions
download_kaggle_dataset(kaggle_dataset, dataset_file)
arxiv_data = load_arxiv_data(dataset_file)



filtered_data = arxiv_data[["id", "title", "abstract", "authors", "categories"]]
filtered_data["abstract"] = filtered_data["abstract"].str.strip()
filtered_data.to_csv("processed_arxiv_metadata.csv", index=False)


# Step 4: Display dataset info
print(arxiv_data.info())
print(arxiv_data.head())

client = MilvusClient("./database_mini.db")
add_hash_to_database("abstracts")


# Example Kaggle file path (to replace with your dataset)
kaggle_file_path = "path_to_kaggle_dataset.csv"
reload_and_lookup(kaggle_file_path, "abstracts")
