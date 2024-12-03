# Imports
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient

from utils.helpers import (
    load_api_key,
    generate_summary_prompt_with_page_content,
    generate_related_work_prompt,
    generate_relevance_evaluation_prompt,
    generate_win_rate_evaluation_prompt
)
from utils.azure_client import AzureClient
from utils.citations import (
    get_arxiv_abstract,
    get_arxiv_citation,
    process_arxiv_paper_with_embeddings,
    filter_citations
)
from utils.long_to_short import (
    select_diverse_papers_with_weighted_similarity,
    select_diverse_pages_for_top_b_papers
)
from dotenv import load_dotenv
import os
import math
import pandas as pd

def run_evaluation(abstract: str, related_work: str) -> pd.DataFrame:
    # abstract = 'Large Language Models have shown impressive per- formance across a wide array of tasks involving both structured and unstructured textual data. More recently, adaptions of these models have drawn attention to their abilities to work with code across different programming languages. On this notion, different benchmarks for code generation, repair, or completion suggest that certain models have programming abilities comparable to or even surpass humans. In this work, we demonstrate that the performance on this benchmark does not translate to the innate ability of humans to appreciate the structural control flow of code. For this purpose, we extract code solutions from the Hu- manEval benchmark, which the relevant models perform very strongly on, and trace their execution path using function calls sampled from the respective test set. Using this dataset, we investigate the ability of 5 state-of-the-art LLMs to match the execution trace and find that, despite the model’s abilities to generate semantically identical code, they possess only limited ability to trace the execution path, especially for traces with increased length. We find that even the top-performing model, Gemini 1.5 Pro can only fully correctly generate the trace of 47% of HumanEval tasks. In addition, we introduce a specific subset for three key structures not, or only contained to a limited extent in Hu- manEval: Recursion, Parallel Processing, and Object Oriented Programming principles, including concepts like Inheritance and Polymorphism. Besides OOP, we show that none of the investigated models achieve an average accuracy of over 5% on the relevant traces. Aggregating these specialized parts with the ubiquitous HumanEval tasks, we present the Benchmark CoCoNUT: Code Control Flow for Navigation Understanding and Testing, which measures a models ability to trace the execu- tion of code upon relevant calls, including advanced structural components. We conclude that the current generation LLMs still need to significantly improve to enhance their code reasoning abilities. We hope our dataset can help researchers bridge this gap in the near future.'
    # related_work = 'Coding Benchmarks for LLMs. Numerous coding bench- marks have been proposed for LLMs in recent years. We summarize and compare a few popular ones here. Open AI HumanEval and MBPP [5] are the most popular code generation benchmarks. Several other benchmarks such as CruxEval [6], LiveCodeBench [7], and Codemind [20] include execution reasoning tasks (in addition to code generation) such as predicting the output of a code snippet given some input. CodeXGlue [21] is another prior dataset consisting of 10 different coding tasks. However, it does not include any execution-related tasks. Runtime Reasoning REval [22] is the most recent work that proposed four execution-related tasks: coverage prediction, program state prediction, execution path prediction (prediction next statement), and output prediction. However, these tasks do not require the LLMs to reason about control flow or different programming structures like the tasks in CoCoNUT. Ma et al. [23] evaluate LLMs on many different tasks, including generating Abstract Syntax Trees, Control Flow, and Call Graphs. However, they limit their investigation of the dynamic behavior of the execution to Equivalent Mutant Detection and Flaky Test Reasoning, which does not directly concern structural understanding abilities. Hooda et al. [24] show that LLMs are vulnerable to different mutations related to control and data flow, as well as type and identifier assignment. A strong understanding of the full execution trace would help build resilience against such approaches. To the best of our knowledge, there is no prior work on evaluating the execution tracing abilities of LLMs, which we investigate in this work. Training LLMs for better execution reasoning. Few recent approaches focused on improving the execution reasoning of LLMs. For instance, Ding et al. developed a new coding dataset augmented with tests and execution traces and trained an LLM, called SemCoder [25]. They showed that such a training strategy elicits better code generation and execution reasoning from the LLM. Ni et al. [15] showed that fine- tuning LLMs on Chain-of-Thought reasoning over execution trace improved the performance of PaLM on the HumanEval and the MBPP benchmarks [5]. However, their approach to tracing mostly consists of variable states for straight-line code, which they insert into the source code as comments instead of control flow reasoning. While they demonstrate that this approach also works without inserting the trace, they note that the model exhibits hallucination issues while adapting the trace into natural language reasoning steps. This naturally motivates enhancing language models’ abilities to directly extract execution representations.'
    IS_THIS_PAPER_ON_ARXIV = False
    breadth = 10
    depth = 2
    diversity = 0

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

    # Clean DB response data
    query_data: list[dict] = query_data[0]
    for obj in query_data:
        obj['embedding'] = obj['entity']['embedding']
        obj.pop('entity')

    # Remove first entry (this is the paper we're searching for). As it has already been published, it will show up with large similarity
    if IS_THIS_PAPER_ON_ARXIV:
        query_data = query_data[1:]

    print(f'Retrieved {len(query_data)} papers from the DB.')

    # Select a longlist of papers
    selected_papers: list[dict] = select_diverse_papers_with_weighted_similarity(
        paper_data=query_data,
        k=3*breadth,
        diversity_weight=diversity
    )

    print(f'Selected {len(selected_papers)} papers for the longlist.')

    relevance_ratings: dict[str, int] = {}
    # calculate average longlist relevance through prompting
    for paper in selected_papers:
        response: str = prompting_client.get_completions(
            generate_relevance_evaluation_prompt(
                source_abstract=abstract,
                target_abstract=get_arxiv_abstract(paper['id'])
            ),
            os.getenv("AZURE_PROMPTING_MODEL_VERSION")
        )
        rating: int = int(response)
        relevance_ratings[paper['id']] = rating

    avg_longlist_rating: float = sum(relevance_ratings.values()) / len(relevance_ratings)
    print(f'Average longlist relevance rating: {avg_longlist_rating}')
    print(relevance_ratings.values())

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

    shortlist_ratings: dict[str, int] = {}
    for obj in relevant_pages:
        arxiv_id = query_data[obj['paper_id']]["id"]
        shortlist_ratings[arxiv_id] = relevance_ratings[arxiv_id]

    avg_shortlist_rating: float = sum(shortlist_ratings.values()) / len(shortlist_ratings)

    print(f'Average shortlist relevance rating: {avg_shortlist_rating}')
    print(shortlist_ratings.values())

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

    win_rate_prompt, order = generate_win_rate_evaluation_prompt(
        source_abstract=abstract,
        source_related_work=related_work,
        target_related_work=related_works_section
    )

    win_rate_response: str = prompting_client.get_completions(
        win_rate_prompt,
        os.getenv("AZURE_PROMPTING_MODEL_VERSION")
    )

    print(win_rate_response)
    # Modify the results to account for reordering
    selected_winner_paper: str = ''
    if win_rate_response == 'Section A':
        print(f'The {order[0]} paper generated related works section was deemed the winner. (source = original)')
        selected_winner_paper = order[0]
    elif win_rate_response == 'Section B':
        print(f'The {order[1]} paper generated related works section was deemed the winner. (source = original)')
        selected_winner_paper = order[1]
    elif win_rate_response == 'Tie':
        print(f'There was a tie.')
        selected_winner_paper = 'tie'


    filtered_citations: list[str] = filter_citations(
        related_works_section=related_works_section,
        citation_strings=[obj['citation'] for obj in relevant_pages]
    )

    print(f'Generated related work section with {len(filtered_citations)} citations.')

    print({
        'related_works': related_works_section,
        'citations': filtered_citations
    })

    return pd.DataFrame(
        data={
            'longlist_length': len(selected_papers),
            'longlist_ratings': [relevance_ratings.values()],
            'shortlist_length': len(relevant_pages),
            'shortlist_ratings': [shortlist_ratings.values()],
            'better_related_works_section': selected_winner_paper,
            'related_works': related_works_section,
            'citations': [filtered_citations]
        }
    )

if __name__ == '__main__':
    # Load environment variables
    load_dotenv()

    input_file = 'sample_data/input.csv'
    output_file = 'sample_data/output.csv'

    # input_df = pd.read_csv(input_file)
    output_df = None
    for i in range(5):
        df = pd.DataFrame(
            data={
                'longlist_length': 1,
                'longlist_ratings': '|'.join([str(i) for i in [1, 2, 3]]),
                'shortlist_length': 1,
                'shortlist_ratings': '|'.join([str(i) for i in [1, 2, 3]]),
                'better_related_works_section': 'abc',
                'related_works': 'def',
                'citations': '\n'.join(['a','b','c'])
            },
            index=['related_works']
        )
        output_df = pd.concat([output_df, df])
    # for row in input_df.iterrows():
    #     row_df: pd.DataFrame = run_evaluation(
    #         abstract=row['abstract'],
    #         related_work=row['related works']
    #     )
    #     if output_df is None:
    #         output_df = row_df
    #     else:
    #         output_df = pd.concat([output_df, row_df])

    output_df.to_csv(output_file, index=False)
