from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
import json
from helpers import load_api_key, generate_summary_prompt
from azure_client import AzureClient
from citations import get_arxiv_abstract

AZURE_ENDPOINT = "cai-project"
AZURE_PROMPTING_MODEL = "gpt-4o"
AZURE_PROMPTING_MODEL_VERSION = "2024-08-01-preview"
KEY_LOCATION = "carl_config.json"

topic_model = BERTopic.load("MaartenGr/BERTopic_ArXiv")
embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
client = MilvusClient('./database.db')
prompting_client = AzureClient(
    endpoint=AZURE_ENDPOINT,
    deployment_id=AZURE_PROMPTING_MODEL,
    api_key=load_api_key(KEY_LOCATION)
)


abstract = 'The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.'
embedded_abstract = embedding_model.encode(abstract)
topic = topic_model.transform(abstract)
topic_id = topic[0][0]

res = client.search(
    collection_name="abstracts",
    data = [embedded_abstract],
    limit = 10,
    # filter = f'topic == {topic_id}',
    search_params = {
        "metric_type": "COSINE",
        "params": {}
    },
    # output_fields = []
)
result = json.dumps(res, indent=4)
print(result)

for obj in result[0]:
    obj_arxiv_abstract = get_arxiv_abstract(obj['id'])
    print(generate_summary_prompt(abstract, obj_arxiv_abstract))
    # response: str = prompting_client.get_completions(generate_summary_prompt(abstract, obj_arxiv_abstract), AZURE_PROMPTING_MODEL_VERSION)
    # print(response)


