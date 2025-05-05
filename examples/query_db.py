from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
import os

abstract = f"""\
TODO....
"""

client = MilvusClient(uri=os.getenv("MILVUS_URI"), token=os.getenv("MILVUS_TOKEN"))
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
embedded_abstract = embedding_model.encode(abstract)

query_data: list[list[dict]] = client.search(
        collection_name="abstracts",
        data=[embedded_abstract],
        limit=60,
        anns_field="embedding",
        search_params={"metric_type": "COSINE", "params": {}},
        output_fields=["embedding"],
)

print(query_data)
client.close()