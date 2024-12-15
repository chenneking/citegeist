import hashlib
import pandas as pd
from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema, Collection
import json

import os
import pandas as pd
from tqdm import tqdm
import time
import torch


from bertopic import BERTopic

def get_metadata(filename):
    with open(filename, 'r') as f:
        for line in f:
            yield line

def compute_embedding(abstract, embedding_model):
    start = time.time()
    res = embedding_model.encode(abstract)
    print("Embedding time: ", time.time()-start)
    return res

def get_embedding(abstract, embedding_model=None):

    if embedding_model is None:
        return torch.zeros(768)
    else:
        return compute_embedding(abstract, embedding_model)


# Add hash to existing database entries
def compute_hash(entry: dict) -> str:
    """Compute a SHA256 hash for a given entry."""
    entry_str = json.dumps(entry, sort_keys=True)
    return hashlib.sha256(entry_str.encode('utf-8')).hexdigest()

# Assuming the database has been loaded into a collection
    


def create_hash_table_from_database(database_file, collection_name="abstracts"):
    print("Creating hash table from database {}...".format(database_file))
    client = MilvusClient(database_file)
    hash_table = {} # id -> hash
    query_start = time.time()
    entries = client.query(collection_name=collection_name, 
                           output_fields=["id", "hash", "embedding"],
                           filter="id != ''")
    print(f"Query time: {time.time()-query_start:.2f} query length: {len(entries)}")
    for entry in tqdm(entries, desc="Processing entries", unit="entry"):
        hash_table[entry["id"]] = entry["hash"]
    return hash_table


def create_hash_table_from_dataset(dataset_file):
    print("Creating hash table from dataset {}...".format(dataset_file))
    hash_table = {} # hash -> id
    total_items = sum(1 for _ in get_metadata(dataset_file))
    progress = tqdm(total=total_items, desc="Processing papers", unit="paper")
    for paper in tqdm(get_metadata(dataset_file)):
        paper = json.loads(paper)
        hash_key = compute_hash(paper)
        hash_table[paper["id"]] = hash_key
        progress.update(1)
    progress.close()
    return hash_table




if __name__ == "__main__":

    # Use the dataset loaded from Kaggle to build the hash table
    dataset_file = "/home/sy774/fall2024/arxiv-metadata-oai-snapshot.json" 
    hash_table_from_dataset = False

    database_file = "/home/sy774/fall2024/database.db"
    collection_name = "arxiv_meta"
    hash_table_from_database = True

    hash_table_file = "/home/sy774/id_hash_table.json"


    if database_file:
        id_hash_table = create_hash_table_from_database(database_file, collection_name=collection_name)
        json.dump(id_hash_table, open(hash_table_file, "w"))
    elif hash_table_from_dataset:
        id_hash_table = create_hash_table_from_dataset(dataset_file)
        json.dump(id_hash_table, open(hash_table_file, "w"))
    

