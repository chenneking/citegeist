# Hash Functions

This folder contains implementations of various hash functions. 

## Main Contents

- `hash_tabling.py`: Run this to create a hash table
    - Change `database_file = "/home/sy774/fall2024/database.db"` to your own file of database path.
    - Change `hash_table_file = "/home/sy774/id_hash_table.json"`  to the path you want to save the hash table
- `hash.reload.py`: After running `hash_tabling.py`, run this to reload and update the arxiv data. Remember to change `hash_table_file = "/home/sy774/id_hash_table.json"`
- `hash_merged.py`: A merged version of the previous two functionality.

## Other contents:

- `hash_redis.py`: Redis version of hash table
- `hash_sqlite.py`: sqlite version of hash table`hash_table_file = "/home/sy774/id_hash_table.json"`
- `hash.py`: pymilvus version of hash table

