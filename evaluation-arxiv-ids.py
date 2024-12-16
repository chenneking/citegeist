import os

import pandas as pd
from dotenv import load_dotenv

from utils.azure_client import AzureClient
from utils.citations import get_arxiv_abstract
from utils.helpers import generate_relevance_evaluation_prompt, load_api_key

load_dotenv()

prompting_client = AzureClient(
    endpoint=os.getenv("AZURE_ENDPOINT"),
    deployment_id=os.getenv("AZURE_PROMPTING_MODEL"),
    api_key=load_api_key(os.getenv("KEY_LOCATION")),
)

papers_df = pd.read_csv('data/papers.csv')
arxiv_id_df = pd.read_csv('data/papers-arxiv-ids.csv')

output_data = []

for i, row in arxiv_id_df.iterrows():
    abstract: str = papers_df['abstract'][i]
    arxiv_ids_source: list[str] = row['arxiv_ids_source'][1:-1].split(', ')
    source_scores = []
    for arxiv_id in arxiv_ids_source:
        abstract2: str = get_arxiv_abstract(arxiv_id)
        prompt: str = generate_relevance_evaluation_prompt(abstract, abstract2)
        response: str = prompting_client.get_completions(
            prompt,
            os.getenv("AZURE_PROMPTING_MODEL_VERSION")
        )
        rating: int = int(response)
        source_scores.append(rating)

    gpt_scores = []
    arxiv_ids_gpt4o_mini: list[str] = row['arxiv_ids_gpt4o_mini'][1:-1].split(', ')
    for arxiv_id in arxiv_ids_gpt4o_mini:
        abstract2: str = get_arxiv_abstract(arxiv_id)
        prompt: str = generate_relevance_evaluation_prompt(abstract, abstract2)
        response: str = prompting_client.get_completions(
            prompt,
            os.getenv("AZURE_PROMPTING_MODEL_VERSION")
        )
        rating: int = int(response)
        gpt_scores.append(rating)

    output_data.append({
        'title': row['title'],
        'source_score': "|".join([str(i) for i in source_scores]),
        'source_length': len(source_scores),
        'gpt4o_score': "|".join([str(i) for i in gpt_scores]),
        'gpt4o_length': len(gpt_scores)
    })

pd.DataFrame(output_data).to_csv('data/output_arxiv-ids.csv', index=False)