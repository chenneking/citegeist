from pymilvus import MilvusClient, DataType
import jsonlines

client = MilvusClient('../database_mini.db')

schema = client.create_schema(
    auto_id=False
)
schema.add_field(field_name='id', datatype=DataType.VARCHAR, max_length=50, is_primary=True)
# schema.add_field(field_name = 'abstract', datatype = DataType.VARCHAR, max_length = 65535)
schema.add_field(field_name='embedding', datatype=DataType.FLOAT_VECTOR, dim=768)
schema.add_field(field_name='hash', datatype=DataType.VARCHAR, max_length=64)
schema.add_field(field_name='topic', datatype=DataType.INT64)

client.create_collection(
    collection_name="abstracts",
    schema=schema
)

print('Starting insertion process')

# List of accepted topics for the mini db
accepted_keys = [79, 77, 76, 63, 62, 60, 40, 30, 19, 16]
key_counts = dict.fromkeys(accepted_keys, 0)
total_count = 0
with jsonlines.open('/Users/carl/Downloads/arxivDrag/processed_data_final_noAbstract.jsonl') as reader:
    for obj in reader:
        if obj['topic'] not in accepted_keys:
            continue
        if total_count % 10000 == 0:
            print(f'Processed {total_count} entries')
        res = client.insert(
            collection_name="abstracts",
            data=[obj]
        )
        total_count += 1
        key_counts[obj['topic']] += 1

print('Completed insertion process')
print(f'Final stats: {key_counts}')

print('Starting indexing')
index_params = MilvusClient.prepare_index_params()

index_params.add_index(
    field_name="embedding",
    metric_type="COSINE",
    index_type="FLAT"
)

client.create_index(
    collection_name="abstracts",
    index_params=index_params
)

index_params.add_index(
    field_name="embedding",
    metric_type="COSINE",
    index_type="FLAT"
)

client.create_index(
    collection_name="abstracts",
    index_params=index_params
)

print('Completed indexing')