# Imports
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
import json
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
    find_most_relevant_pages,
)
from utils.long_to_short import (
    extract_most_relevant_pages,
    select_diverse_papers_with_weighted_similarity,
    extract_most_relevant_pages_for_each_paper,
    select_diverse_pages_for_top_b_papers
)
from dotenv import load_dotenv
import os

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
                page_text_to_be_cited=text_segments
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
            data=relevant_pages
        ),
        os.getenv("AZURE_PROMPTING_MODEL_VERSION")
    )

    print('Generated related work section.')

    return {
        'related_works': related_works_section,
        'citations': [obj['citation'] for obj in relevant_pages]
    }

# if __name__ == "__main__":
#     print(generate_related_work(
#         "Large Language Models have shown impressive performance across a wide array of tasks involving both structured and unstructured textual data. More recently, adaptions of these models have drawn attention to their abilities to work with code across different programming languages. On this notion, different benchmarks for code generation, repair, or completion suggest that certain models have programming abilities comparable to or even surpass humans. In this work, we demonstrate that the performance on this benchmark does not translate to the innate ability of humans to appreciate the structural control flow of code. For this purpose, we extract code solutions from the Hu- manEval benchmark, which the relevant models perform very strongly on, and trace their execution path using function calls sampled from the respective test set. Using this dataset, we investigate the ability of 5 state-of-the-art LLMs to match the execution trace and find that, despite the model’s abilities to generate semantically identical code, they possess only limited ability to trace the execution path, especially for traces with increased length. We find that even the top-performing model, Gemini 1.5 Pro can only fully correctly generate the trace of 47% of HumanEval tasks. In addition, we introduce a specific subset for three key structures not, or only contained to a limited extent in Hu- manEval: Recursion, Parallel Processing, and Object Oriented Programming principles, including concepts like Inheritance and Polymorphism. Besides OOP, we show that none of the investigated models achieve an average accuracy of over 5% on the relevant traces. Aggregating these specialized parts with the ubiquitous HumanEval tasks, we present the Benchmark CoCoNUT: Code Control Flow for Navigation Understanding and Testing, which measures a models ability to trace the execu- tion of code upon relevant calls, including advanced structural components. We conclude that the current generation LLMs still need to significantly improve to enhance their code reasoning abilities. We hope our dataset can help researchers bridge this gap in the near future.",
#         10,
#         2,
#         0.1
#     ))

