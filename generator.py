# Imports
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
from utils.helpers import (
    load_api_key,
    generate_summary_prompt_with_page_content,
    generate_related_work_prompt,
)
from utils.azure_client import AzureClient
from utils.citations import (
    get_arxiv_abstract,
    get_arxiv_citation,
    process_arxiv_paper_with_embeddings,
)
from utils.long_to_short import (
    select_diverse_papers_with_weighted_similarity,
    select_diverse_pages_for_top_b_papers
)
from dotenv import load_dotenv
import os
import math

# Load environment variables
load_dotenv()

def generate_related_work(abstract: str, breadth: int, depth: int, diversity: float) -> dict[str, str | list[str]]:
    # Initialize clients
    topic_model = BERTopic.load("MaartenGr/BERTopic_ArXiv")
    embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    client = MilvusClient("./database.db")
    prompting_client = AzureClient(
        endpoint=os.getenv("AZURE_ENDPOINT"),
        deployment_id=os.getenv("AZURE_PROMPTING_MODEL"),
        api_key=load_api_key(os.getenv("KEY_LOCATION")),
    )

    embedded_abstract = embedding_model.encode(abstract)
    topic = topic_model.transform(abstract)
    topic_id = topic[0][0]

    # Query Milvus Vector DB
    query_data: list[list[dict]] = client.search(
        collection_name="abstracts",
        data=[embedded_abstract],
        limit=6*breadth,
        anns_field="embedding",
        # filter = f'topic == {topic_id}',
        search_params={"metric_type": "COSINE", "params": {}},
        output_fields=["embedding"],
    )

    print(f'Retrieved {len(query_data)} papers from the DB.')

    # Clean DB response data
    query_data: list[dict] = query_data[0]
    for obj in query_data:
        obj['embedding'] = obj['entity']['embedding']
        obj.pop('entity')

    # Select a longlist of papers
    selected_papers: list[dict] = select_diverse_papers_with_weighted_similarity(
        paper_data=query_data,
        k=3*breadth,
        diversity_weight=diversity
    )

    print(f'Selected {len(selected_papers)} papers for the longlist.')

    # Generate embeddings of each page of every paper in the longlist
    page_embeddings: list[list[dict]] = []
    for paper in selected_papers:
        arxiv_id = paper["id"]
        result = process_arxiv_paper_with_embeddings(arxiv_id, topic_model)
        if result:
            page_embeddings.append(result)

    print(f'Generated page embeddings for {len(page_embeddings)} papers.')

    # Generate shortlist of papers (at most k pages per paper, at most b papers in total)
    relevant_pages: list[dict] = select_diverse_pages_for_top_b_papers(
        paper_embeddings=page_embeddings,
        input_string=abstract,
        topic_model=topic_model,
        k=depth,
        b=breadth,
        diversity_weight=diversity,
        skip_first=False
    )

    print(f'Selected {len(relevant_pages)} papers for the shortlist.')

    # Generate summaries for individual papers (taking all relevant pages into account)
    for obj in relevant_pages:
        # Because paper_id != arXiv_id -> retrieve arXiv id/
        arxiv_id = query_data[obj['paper_id']]["id"]
        arxiv_abstract = get_arxiv_abstract(arxiv_id)
        text_segments = obj["text"]
        response: str = prompting_client.get_completions(
            generate_summary_prompt_with_page_content(
                abstract_source_paper=abstract,
                abstract_to_be_cited=arxiv_abstract,
                page_text_to_be_cited=text_segments,
                sentence_count=5
            ),
            os.getenv("AZURE_PROMPTING_MODEL_VERSION")
        )
        obj["summary"] = response
        obj["citation"] = get_arxiv_citation(arxiv_id)

    print('Generated summaries of papers (and their pages).')

    # Generate the final related works section text
    related_works_section: str = prompting_client.get_completions(
        generate_related_work_prompt(
            source_abstract=abstract,
            data=relevant_pages,
            paragraph_count=math.ceil(breadth/2),
            add_summary=False
        ),
        os.getenv("AZURE_PROMPTING_MODEL_VERSION")
    )

    print('Generated related work section.')

    return {
        'related_works': related_works_section,
        'citations': [obj['citation'] for obj in relevant_pages]
    }

