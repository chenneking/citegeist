# Imports
import math
import os
from typing import Callable

from bertopic import BERTopic
from dotenv import load_dotenv
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

from citegeist.utils.citations import (
    filter_citations,
    get_arxiv_abstract,
    get_arxiv_citation,
    process_arxiv_paper_with_embeddings,
)
from citegeist.utils.filtering import (
    select_diverse_pages_for_top_b_papers,
    select_diverse_papers_with_weighted_similarity,
)
from citegeist.utils.helpers import load_api_key
from citegeist.utils.llm_clients import create_client
from citegeist.utils.prompts import (
    generate_question_answer_prompt,
    generate_related_work_prompt,
    generate_summary_prompt_question_with_page_content,
    generate_summary_prompt_with_page_content,
)

# Load environment variables
load_dotenv()

# Configure the LLM client provider from environment variables
DEFAULT_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "azure")


class Generator:
    """Main generator class for Citegeist."""

    def __init__(
        self,
        llm_provider: str,  # defaults to: azure
        database_path: str,  # path to local milvus DB file
        sentence_embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        topic_model_name: str = "MaartenGr/BERTopic_ArXiv",
        **llm_kwargs,
    ):
        """
        Initialize the Generator with configuration.

        Args:
            llm_provider: LLM provider name ('azure', 'openai', 'anthropic').
                          Falls back to environment variable LLM_PROVIDER, then to 'azure'
            sentence_embedding_model_name: Name of the sentence transformer embedding model
            topic_model_name: Name of the BERTopic model
            database_path: Path to the Milvus database
            **llm_kwargs: Provider-specific configuration arguments for the LLM client
        """
        # Initialize core models
        self.topic_model = BERTopic.load(topic_model_name)
        self.sentence_embedding_model = SentenceTransformer(sentence_embedding_model_name)
        self.db_client = MilvusClient(database_path)

        # Set up LLM client
        self.llm_provider = llm_provider or DEFAULT_LLM_PROVIDER

        # Create LLM client
        self.llm_client = create_client(self.llm_provider, **llm_kwargs)

        # Store API version for Azure compatibility
        self.api_version = os.getenv("AZURE_API_VERSION", "2023-05-15")

    def generate_related_work(
        self,
        abstract: str,
        breadth: int,
        depth: int,
        diversity: float,
        status_callback: Callable = None,
    ) -> dict[str, str | list[str]]:
        """
        Generate a related work section based on an abstract.

        Args:
            abstract: The input abstract text
            breadth: Number of papers to consider
            depth: Number of pages to extract from each paper
            diversity: Diversity factor for paper selection (0-1)
            status_callback: Callback function that will update jobs according to the function progress

        Returns:
            Dictionary with 'related_works' text and 'citations' list
        """
        return generate_related_work(
            abstract,
            breadth,
            depth,
            diversity,
            self.topic_model,
            self.sentence_embedding_model,
            self.db_client,
            self.llm_client,
            self.api_version,
            status_callback,
        )

    def generate_related_work_from_paper(
        self,
        pages: list[str],
        breadth: int,
        depth: int,
        diversity: float,
        status_callback: Callable = None,
    ) -> dict[str, str | list[str]]:
        """
        Generate a related work section based on a full paper.

        Args:
            pages: List of paper pages
            breadth: Number of papers to consider
            depth: Number of pages to extract from each paper
            diversity: Diversity factor for paper selection (0-1)
            status_callback: Callback function that will update jobs according to the function progress

        Returns:
            Dictionary with 'related_works' text and 'citations' list
        """
        return generate_related_work_from_paper(
            pages,
            breadth,
            depth,
            diversity,
            self.topic_model,
            self.sentence_embedding_model,
            self.db_client,
            self.llm_client,
            self.api_version,
            status_callback,
        )

    def generate_answer_to_scientific_question(
        self,
        question: str,
        breadth: int,
        depth: int,
        diversity: float,
        status_callback: Callable = None,
    ) -> dict[str, str | list[str]]:
        """
        Generate an answer to a scientific question.

        Args:
            question: The input question text
            breadth: Number of papers to consider
            depth: Number of pages to extract from each paper
            diversity: Diversity factor for paper selection (0-1)
            status_callback: Callback function that will update jobs according to the function progress

        Returns:
            Dictionary with 'question_answer' text and 'citations' list
        """
        return generate_answer_to_scientific_question(
            question,
            breadth,
            depth,
            diversity,
            self.topic_model,
            self.sentence_embedding_model,
            self.db_client,
            self.llm_client,
            self.api_version,
            status_callback,
        )

    # TODO: remove this in final version. just used for testing of webapp.
    async def dummy(self, status_callback: Callable) -> dict[str, str | list[str]]:
        time = 3.0
        import asyncio

        status_callback(1, "Initializing.")
        await asyncio.sleep(time)
        status_callback(2, "Querying Vector DB for matches.")
        await asyncio.sleep(time)
        status_callback(3, "Retrieved 60 papers from the DB.")
        await asyncio.sleep(time)
        status_callback(4, "Selected 40 papers for the longlist.")
        await asyncio.sleep(time)
        status_callback(5, "Generated page embeddings for 39 papers.")
        await asyncio.sleep(time)
        status_callback(6, "Selected 15 papers for the shortlist.")
        await asyncio.sleep(time)
        status_callback(7, "Generated summaries of papers (and their pages).")
        await asyncio.sleep(time)
        status_callback(8, "Generated related work section with 10 citations.")

        return {"related_works": "testing1223", "citations": ["a", "b", "c"]}


def generate_related_work(
    abstract: str,
    breadth: int,
    depth: int,
    diversity: float,
    topic_model=None,
    embedding_model=None,
    client=None,
    llm_client=None,
    api_version=None,
    status_callback=None,
) -> dict[str, str | list[str]]:
    """
    Generate a related work section based on an abstract.

    Args:
        abstract: The input abstract text
        breadth: Number of papers to consider
        depth: Number of pages to extract from each paper
        diversity: Diversity factor for paper selection (0-1)
        topic_model: Optional pre-initialized BERTopic model
        embedding_model: Optional pre-initialized SentenceTransformer model
        client: Optional pre-initialized MilvusClient
        llm_client: Optional pre-initialized LLM client
        api_version: API version (for Azure compatibility)
        status_callback: Optional callback function that updates job status

    Returns:
        Dictionary with 'related_works' text and 'citations' list
    """
    if status_callback:
        status_callback(1, "Initializing")

    # Initialize models and clients if not provided
    if topic_model is None:
        topic_model = BERTopic.load("MaartenGr/BERTopic_ArXiv")

    if embedding_model is None:
        embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    if client is None:
        client = MilvusClient("./database.db")

    if llm_client is None:
        # Create default Azure client for backward compatibility
        llm_client = create_client(
            "azure",
            endpoint=os.getenv("AZURE_ENDPOINT"),
            deployment_id=os.getenv("AZURE_PROMPTING_MODEL"),
            api_key=load_api_key(os.getenv("KEY_LOCATION")),
        )

    if api_version is None:
        api_version = os.getenv("AZURE_API_VERSION", "2023-05-15")

    embedded_abstract = embedding_model.encode(abstract)
    # topic = topic_model.transform(abstract)
    # topic_id = topic[0][0]

    # Query Milvus Vector DB
    if status_callback:
        status_callback(2, "Querying Vector DB for matches (this may take a while)")

    query_data: list[list[dict]] = client.search(
        collection_name="abstracts",
        data=[embedded_abstract],
        limit=6 * breadth,
        anns_field="embedding",
        # filter = f'topic == {topic_id}',
        search_params={"metric_type": "COSINE", "params": {}},
        output_fields=["embedding"],
    )

    if status_callback:
        status_callback(3, f"Retrieved {len(query_data)} papers from the DB")

    # Clean DB response data
    query_data: list[dict] = query_data[0]
    for obj in query_data:
        obj["embedding"] = obj["entity"]["embedding"]
        obj.pop("entity")

    # Select a longlist of papers
    selected_papers: list[dict] = select_diverse_papers_with_weighted_similarity(
        paper_data=query_data, k=3 * breadth, diversity_weight=diversity
    )

    if status_callback:
        status_callback(4, f"Selected {len(selected_papers)} papers for the longlist")

    # Generate embeddings of each page of every paper in the longlist
    page_embeddings: list[list[dict]] = []
    for paper in selected_papers:
        arxiv_id = paper["id"]
        result = process_arxiv_paper_with_embeddings(arxiv_id, topic_model)
        if result:
            page_embeddings.append(result)

    if status_callback:
        status_callback(5, f"Generated page embeddings for {len(page_embeddings)} papers")

    # Generate shortlist of papers (at most k pages per paper, at most b papers in total)
    relevant_pages: list[dict] = select_diverse_pages_for_top_b_papers(
        paper_embeddings=page_embeddings,
        input_string=abstract,
        topic_model=topic_model,
        k=depth,
        b=breadth,
        diversity_weight=diversity,
        skip_first=False,
    )

    if status_callback:
        status_callback(6, f"Selected {len(relevant_pages)} papers for the shortlist")

    # Generate summaries for individual papers (taking all relevant pages into account)
    for obj in relevant_pages:
        # Because paper_id != arXiv_id -> retrieve arXiv id/
        arxiv_id = query_data[obj["paper_id"]]["id"]
        arxiv_abstract = get_arxiv_abstract(arxiv_id)
        text_segments = obj["text"]

        # Create prompt
        prompt = generate_summary_prompt_with_page_content(
            abstract_source_paper=abstract,
            abstract_to_be_cited=arxiv_abstract,
            page_text_to_be_cited=text_segments,
            sentence_count=5,
        )

        # Use the appropriate LLM client based on provider
        response: str = llm_client.get_completion(prompt)
        obj["summary"] = response
        obj["citation"] = get_arxiv_citation(arxiv_id)

    if status_callback:
        status_callback(7, "Generated summaries of papers (and their pages)")

    # Generate the final related works section text
    prompt = generate_related_work_prompt(
        source_abstract=abstract, data=relevant_pages, paragraph_count=math.ceil(breadth / 2), add_summary=False
    )

    # Use the appropriate LLM client based on provider
    related_works_section: str = llm_client.get_completion(prompt)

    filtered_citations: list[str] = filter_citations(
        related_works_section=related_works_section, citation_strings=[obj["citation"] for obj in relevant_pages]
    )

    if status_callback:
        status_callback(8, f"Generated related work section with {len(filtered_citations)} citations")

    return {"related_works": related_works_section, "citations": filtered_citations}


def generate_answer_to_scientific_question(
    question: str,
    breadth: int,
    depth: int,
    diversity: float,
    topic_model=None,
    embedding_model=None,
    client=None,
    llm_client=None,
    api_version=None,
    status_callback=None,
) -> dict[str, str | list[str]]:
    """
    Generate an answer to a scientific question with citations.

    Args:
        question: The input question
        breadth: Number of papers to consider
        depth: Number of pages to extract from each paper
        diversity: Diversity factor for paper selection (0-1)
        topic_model: Optional pre-initialized BERTopic model
        embedding_model: Optional pre-initialized SentenceTransformer model
        client: Optional pre-initialized MilvusClient
        llm_client: Optional pre-initialized LLM client
        api_version: API version (for Azure compatibility)

    Returns:
        Dictionary with 'question_answer' text and 'citations' list
    """
    if status_callback:
        status_callback(1, "Initializing.")
    print("Initializing.")
    # Initialize models and clients if not provided
    if topic_model is None:
        topic_model = BERTopic.load("MaartenGr/BERTopic_ArXiv")

    if embedding_model is None:
        embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    if client is None:
        client = MilvusClient("./database.db")

    if llm_client is None:
        # Create default Azure client for backward compatibility
        llm_client = create_client(
            "azure",
            endpoint=os.getenv("AZURE_ENDPOINT"),
            deployment_id=os.getenv("AZURE_PROMPTING_MODEL"),
            api_key=load_api_key(os.getenv("KEY_LOCATION")),
        )

    if api_version is None:
        api_version = os.getenv("AZURE_API_VERSION", "2023-05-15")

    embedded_abstract = embedding_model.encode(question)
    # topic = topic_model.transform(question)
    # topic_id = topic[0][0]

    # Query Milvus Vector DB
    if status_callback:
        status_callback(2, "Querying Vector DB for matches (this may take a while)")

    query_data: list[list[dict]] = client.search(
        collection_name="abstracts",
        data=[embedded_abstract],
        limit=6 * breadth,
        anns_field="embedding",
        # filter = f'topic == {topic_id}',
        search_params={"metric_type": "COSINE", "params": {}},
        output_fields=["embedding"],
    )

    if status_callback:
        status_callback(3, f"Retrieved {len(query_data)} papers from the DB")

    # Clean DB response data
    query_data: list[dict] = query_data[0]
    for obj in query_data:
        obj["embedding"] = obj["entity"]["embedding"]
        obj.pop("entity")

    # Select a longlist of papers
    selected_papers: list[dict] = select_diverse_papers_with_weighted_similarity(
        paper_data=query_data, k=3 * breadth, diversity_weight=diversity
    )

    if status_callback:
        status_callback(4, f"Selected {len(selected_papers)} papers for the longlist")

    # Generate embeddings of each page of every paper in the longlist
    page_embeddings: list[list[dict]] = []
    for paper in selected_papers:
        arxiv_id = paper["id"]
        result = process_arxiv_paper_with_embeddings(arxiv_id, topic_model)
        if result:
            page_embeddings.append(result)

    if status_callback:
        status_callback(5, f"Generated page embeddings for {len(page_embeddings)} papers")

    # Generate shortlist of papers (at most k pages per paper, at most b papers in total)
    relevant_pages: list[dict] = select_diverse_pages_for_top_b_papers(
        paper_embeddings=page_embeddings,
        input_string=question,
        topic_model=topic_model,
        k=depth,
        b=breadth,
        diversity_weight=diversity,
        skip_first=False,
    )

    if status_callback:
        status_callback(6, f"Selected {len(relevant_pages)} papers for the shortlist")

    # Generate summaries for individual papers (taking all relevant pages into account)
    for obj in relevant_pages:
        # Because paper_id != arXiv_id -> retrieve arXiv id/
        arxiv_id = query_data[obj["paper_id"]]["id"]
        arxiv_abstract = get_arxiv_abstract(arxiv_id)
        text_segments = obj["text"]
        # Create prompt
        prompt = generate_summary_prompt_question_with_page_content(
            question=question, abstract_to_be_considered=arxiv_abstract, page_text_to_be_cited=text_segments
        )

        # Use the appropriate LLM client
        response: str = llm_client.get_completion(prompt)
        obj["summary"] = response
        obj["citation"] = get_arxiv_citation(arxiv_id)

    if status_callback:
        status_callback(7, "Generated summaries of papers (and their pages)")

    # Generate the final question answer
    prompt = generate_question_answer_prompt(question=question, data=relevant_pages)

    # Use the appropriate LLM client
    question_answer: str = llm_client.get_completion(prompt)

    filtered_citations: list[str] = filter_citations(
        related_works_section=question_answer, citation_strings=[obj["citation"] for obj in relevant_pages]
    )

    if status_callback:
        status_callback(8, f"Generated answer to question with {len(filtered_citations)} citations")

    return {"question_answer": question_answer, "citations": filtered_citations}


def generate_related_work_from_paper(
    pages: list[str],
    breadth: int,
    depth: int,
    diversity: float,
    topic_model=None,
    embedding_model=None,
    client=None,
    llm_client=None,
    api_version=None,
    status_callback=None,
) -> dict[str, str | list[str]]:
    """
    Generate a related work section based on full paper pages.

    Args:
        pages: List of paper pages as text
        breadth: Number of papers to consider
        depth: Number of pages to extract from each paper
        diversity: Diversity factor for paper selection (0-1)
        topic_model: Optional pre-initialized BERTopic model
        embedding_model: Optional pre-initialized SentenceTransformer model
        client: Optional pre-initialized MilvusClient
        llm_client: Optional pre-initialized LLM client
        api_version: API version (for Azure compatibility)
        status_callback: Optional callback function that updates job status

    Returns:
        Dictionary with 'related_works' text and 'citations' list
    """
    if status_callback:
        status_callback(1, "Initializing.")
    print("Initializing.")
    # Initialize models and clients if not provided
    if topic_model is None:
        topic_model = BERTopic.load("MaartenGr/BERTopic_ArXiv")

    if embedding_model is None:
        embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    if client is None:
        client = MilvusClient("./database.db")

    if llm_client is None:
        # Create default Azure client for backward compatibility
        llm_client = create_client(
            "azure",
            endpoint=os.getenv("AZURE_ENDPOINT"),
            deployment_id=os.getenv("AZURE_PROMPTING_MODEL"),
            api_key=load_api_key(os.getenv("KEY_LOCATION")),
        )

    if api_version is None:
        api_version = os.getenv("AZURE_API_VERSION", "2023-05-15")

    # Create embeddings for all pages
    page_embeddings = [embedding_model.encode(page) for page in pages]

    # Query Milvus Vector DB for each page
    if status_callback:
        status_callback(2, "Querying Vector DB for matches (this may take a while)")

    all_query_data: list[list[dict]] = []
    for embedding in page_embeddings:
        query_result = client.search(
            collection_name="abstracts",
            data=[embedding],
            limit=6 * breadth,
            anns_field="embedding",
            # filter = f'topic == {topic_id}',  # Could potentially use topic_ids here
            search_params={"metric_type": "COSINE", "params": {}},
            output_fields=["embedding"],
        )
        all_query_data.extend(query_result)

    if status_callback:
        status_callback(3, f"Retrieved papers from DB for {len(all_query_data)} pages")

    # Aggregate similarity scores for papers that appear multiple times
    paper_scores: dict[str, float] = {}
    paper_data: dict[str, dict] = {}

    for page_results in all_query_data:
        for result in page_results:
            paper_id = result["id"]
            similarity_score = result["distance"]  # Assuming this is the similarity score

            if paper_id in paper_scores:
                paper_scores[paper_id] += similarity_score
            else:
                paper_scores[paper_id] = similarity_score
                paper_data[paper_id] = {"id": paper_id, "embedding": result["entity"]["embedding"]}

    # Convert aggregated results back to format expected by select_diverse_papers
    # Sort papers by aggregated score and take top 6*breadth papers
    top_paper_ids = sorted(paper_scores.items(), key=lambda x: x[1], reverse=True)[: 6 * breadth]

    # Convert back to original format expected by select_diverse_papers
    # Each entry should be a list with one dict per query result
    aggregated_query_data = [
        {"id": paper_id, "embedding": paper_data[paper_id]["embedding"], "distance": score}
        for paper_id, score in top_paper_ids
    ]

    # Select a longlist of papers using aggregated scores
    selected_papers: list[dict] = select_diverse_papers_with_weighted_similarity(
        paper_data=aggregated_query_data, k=3 * breadth, diversity_weight=diversity
    )

    if status_callback:
        status_callback(4, f"Selected {len(selected_papers)} papers for the longlist")

    # Generate embeddings of each page of every paper in the longlist
    page_embeddings_papers: list[list[dict]] = []
    for paper in selected_papers:
        arxiv_id = paper["id"]
        result = process_arxiv_paper_with_embeddings(arxiv_id, topic_model)
        if result:
            page_embeddings_papers.append(result)

    if status_callback:
        status_callback(5, f"Generated page embeddings for {len(page_embeddings)} papers")

    # Generate shortlist of papers using first page as reference
    # (you might want to modify this to consider all input pages)
    relevant_pages: list[dict] = select_diverse_pages_for_top_b_papers(
        paper_embeddings=page_embeddings_papers,
        input_string=pages[0],  # Using first page as reference
        topic_model=topic_model,
        k=depth,
        b=breadth,
        diversity_weight=diversity,
        skip_first=False,
    )

    if status_callback:
        status_callback(6, f"Selected {len(relevant_pages)} papers for the shortlist")

    # Generate summaries for individual papers
    for obj in relevant_pages:
        arxiv_id = aggregated_query_data[obj["paper_id"]]["id"]
        arxiv_abstract = get_arxiv_abstract(arxiv_id)
        text_segments = obj["text"]
        # Create prompt
        prompt = generate_summary_prompt_with_page_content(
            abstract_source_paper=pages[0],  # Using first page as reference
            abstract_to_be_cited=arxiv_abstract,
            page_text_to_be_cited=text_segments,
            sentence_count=5,
        )

        # Use the appropriate LLM client
        response: str = llm_client.get_completion(prompt)
        obj["summary"] = response
        obj["citation"] = get_arxiv_citation(arxiv_id)

    if status_callback:
        status_callback(7, "Generated summaries of papers (and their pages)")

    # Generate the final related works section text
    prompt = generate_related_work_prompt(
        source_abstract=pages[0],  # Using first page as reference
        data=relevant_pages,
        paragraph_count=math.ceil(breadth / 2),
        add_summary=False,
    )

    # Use the appropriate LLM client
    related_works_section: str = llm_client.get_completion(prompt)

    filtered_citations: list[str] = filter_citations(
        related_works_section=related_works_section, citation_strings=[obj["citation"] for obj in relevant_pages]
    )

    if status_callback:
        status_callback(8, f"Generated related work section with {len(filtered_citations)} citations")

    return {"related_works": related_works_section, "citations": filtered_citations}
