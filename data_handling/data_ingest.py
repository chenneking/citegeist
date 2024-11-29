from pymilvus import MilvusClient, DataType
import jsonlines

# Initiate Milvus Client
client = MilvusClient("../database.db")

# Define Milvus DB Schema
schema = client.create_schema(auto_id=False)
schema.add_field(
    field_name="id", datatype=DataType.VARCHAR, max_length=50, is_primary=True
)
# schema.add_field(field_name = 'abstract', datatype = DataType.VARCHAR, max_length = 65535)
schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=768)
schema.add_field(field_name="hash", datatype=DataType.VARCHAR, max_length=64)
schema.add_field(field_name="topic", datatype=DataType.INT64)

client.create_collection(collection_name="abstracts", schema=schema)

# Read in the entire jsonl file data and insert it into the DB
count = 0
with jsonlines.open(
    "/Users/carl/Downloads/arxivDrag/processed_data_final_noAbstract.jsonl"
) as reader:
    for obj in reader:
        if count % 10000 == 0:
            print(f"Processed {count} entries")
        res = client.insert(collection_name="abstracts", data=[obj])
        count += 1

# Add index on the embedding vector values
index_params = MilvusClient.prepare_index_params()

index_params.add_index(field_name="embedding", metric_type="COSINE", index_type="FLAT")

client.create_index(collection_name="abstracts", index_params=index_params)
