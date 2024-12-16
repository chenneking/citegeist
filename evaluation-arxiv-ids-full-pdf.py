import os

import pandas as pd
from dotenv import load_dotenv

from utils.azure_client import AzureClient
from utils.citations import get_arxiv_abstract
from utils.helpers import generate_relevance_evaluation_prompt, load_api_key, generate_related_work_score_prompt

load_dotenv()

prompting_client = AzureClient(
    endpoint=os.getenv("AZURE_ENDPOINT"),
    deployment_id=os.getenv("AZURE_PROMPTING_MODEL"),
    api_key=load_api_key(os.getenv("KEY_LOCATION")),
)

OUTPUT_RELATED_WORK_RELEVANCE_SCORE_PATH = 'data/output_related_work_scores.csv'

papers_df = pd.read_csv('data/papers.csv')
output_df = pd.read_csv('data/output.csv')
output_full_pdf_df = pd.read_csv('data/output_full_pdf.csv')

related_works_score_df = pd.read_csv('data/output_related_work_scores.csv')

output_related_work_scores_data = []

for i, row in papers_df.iterrows():
    abstract = papers_df['abstract'][i]
    full_pdf_related_work = output_full_pdf_df['related_works'][i]

    prompt_full_pdf: str = generate_related_work_score_prompt(abstract, full_pdf_related_work)

    result_full_pdf: str = prompting_client.get_completions(
        prompt_full_pdf,
        os.getenv("AZURE_PROMPTING_MODEL_VERSION")
    )
    rating_full_pdf: int = int(result_full_pdf)
    output_related_work_scores_data.append(rating_full_pdf)

related_works_score_df['score_ours_full_pdf'] = output_related_work_scores_data

related_works_score_df.to_csv(OUTPUT_RELATED_WORK_RELEVANCE_SCORE_PATH, index=False)




