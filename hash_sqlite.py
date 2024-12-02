import sqlite3
import hashlib
import pandas as pd
import json

DB_FILE = "database_mini.db"

def compute_hash(entry: dict) -> str:
    """Compute a SHA256 hash for a given entry."""
    entry_str = json.dumps(entry, sort_keys=True)  # Serialize entry for consistency
    return hashlib.sha256(entry_str.encode("utf-8")).hexdigest()


# def create_database():
#     """Create SQLite database and table if not exists."""
#     conn = sqlite3.connect(DB_FILE)
#     cursor = conn.cursor()
#     cursor.execute("""
#         CREATE TABLE IF NOT EXISTS abstracts (
#             id INTEGER PRIMARY KEY,
#             data TEXT NOT NULL,
#             hash TEXT UNIQUE NOT NULL
#         )
#     """)
#     conn.commit()
#     conn.close()



def migrate_table():
    conn = sqlite3.connect("database_mini.db")
    cursor = conn.cursor()
    
    # Create a new table with the correct schema
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS abstracts_new (
            id INTEGER PRIMARY KEY,
            data TEXT NOT NULL,
            hash TEXT UNIQUE NOT NULL
        )
    """)
    
    # Copy data from the old table to the new table
    cursor.execute("SELECT id, data FROM abstracts")
    rows = cursor.fetchall()
    for row in rows:
        entry_id, data = row
        # Compute the hash for existing data
        hash_value = hashlib.sha256(data.encode('utf-8')).hexdigest()
        cursor.execute("INSERT INTO abstracts_new (id, data, hash) VALUES (?, ?, ?)", 
                       (entry_id, data, hash_value))
    
    # Drop the old table
    cursor.execute("DROP TABLE abstracts")
    
    # Rename the new table to the old table's name
    cursor.execute("ALTER TABLE abstracts_new RENAME TO abstracts")
    
    print("Table 'abstracts' migrated successfully.")
    
    conn.commit()
    conn.close()

migrate_table()




def add_hash_to_database(data: list):
    """Add hash and data to the SQLite database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Check if the column exists
    cursor.execute("PRAGMA table_info(abstracts)")
    columns = [column[1] for column in cursor.fetchall()]
    if "hash" not in columns:
        cursor.execute("ALTER TABLE abstracts ADD COLUMN hash TEXT UNIQUE")
        print("Column 'hash' added successfully.")
    else:
        print("Column 'hash' already exists.")



    for entry in data:
        entry_hash = compute_hash(entry)
        entry_json = json.dumps(entry)
        cursor.execute("INSERT INTO abstracts (data, hash) VALUES (?, ?)", (entry_json, entry_hash))

    conn.commit()
    conn.close()

# Step 4: Reload dataset and perform lookups
def reload_and_lookup(kaggle_file: str):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Load new data from Kaggle
    new_data = pd.read_csv(kaggle_file)
    for _, row in new_data.iterrows():
        metadata = row.to_dict()
        metadata_hash = compute_hash(metadata)

        # Lookup the hash in the database
        cursor.execute("SELECT data FROM abstracts WHERE hash = ?", (metadata_hash,))
        match = cursor.fetchone()
        if match:
            print(f"Match found for metadata: {metadata}")
        else:
            print(f"No match found for metadata: {metadata}")

    conn.close()

# Example usage
if __name__ == "__main__":
    # Example data for initial database population
    initial_data = [
        {"title": "Example Paper 1", "author": "Author A", "year": 2023},
        {"title": "Example Paper 2", "author": "Author B", "year": 2024},
    ]
    # create_database()
    migrate_table()
    add_hash_to_database(initial_data)

    # Path to Kaggle dataset
    kaggle_file_path = "path_to_kaggle_dataset.csv"
    reload_and_lookup(kaggle_file_path)
