import json
from utils.citations import process_arxiv_paper_with_embeddings
from bertopic import BERTopic
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def extract_most_relevant_pages(
    paper_embeddings: list[object],
    input_string: str,
    topic_model: BERTopic,
    top_k: int = 5,
    skip_first: bool = False,
) -> list[dict]:
    """
    Extracts the most relevant pages based on cosine similarity with an input string.

    Args:
        paper_embeddings (list): A list of papers, each containing dictionaries with "text" and "embedding".
        input_string (str): The string to compare against.
        topic_model (BERTopic): The BERTopic model for generating embeddings.
        top_k (int): The number of most relevant pages to return.
        skip_first (bool): Whether to skip the first page of the input string.

    Returns:
        list: A list of dictionaries with "paper_id", "page_number", "text", and "similarity".
    """
    # Encode the input string to get its embedding
    embedding_model = topic_model.embedding_model
    input_embedding = embedding_model.embedding_model.encode(input_string)

    results = []

    for paper_idx, paper in enumerate(paper_embeddings):
        paper_id = (
            paper_idx  # Replace with actual paper ID if available in `paper_embeddings`
        )
        for page_number, page_data in enumerate(paper):
            if skip_first and page_number == 0:
                continue
            page_text = page_data["text"]
            page_embedding = page_data["embedding"]

            # Compute cosine similarity
            similarity = cosine_similarity([input_embedding], [page_embedding])[0][0]

            # Append results
            results.append(
                {
                    "paper_id": paper_id,  # Or use a specific ID if available
                    "page_number": page_number + 1,  # Pages are 1-indexed
                    "text": page_text,
                    "similarity": similarity,
                }
            )

    # Sort results by similarity score in descending order
    results = sorted(results, key=lambda x: x["similarity"], reverse=True)

    # Return the top-k most relevant pages
    return results[:top_k]


def select_diverse_papers_with_precomputed_distances(
    paper_data: list, k: int
) -> list[str]:
    """
    Selects `k` papers that maximize diversity while minimizing distance to the input paper.

    Args:
        paper_data (list): List of dictionaries, where each dictionary represents a paper and contains:
                           - "id": Paper ID
                           - "distance": Similarity to the input paper
                           - "embedding": Embedding of the paper
        k (int): Number of papers to select.

    Returns:
        list: IDs of the selected papers.
    """
    # Extract distances and embeddings
    distances = np.array([paper["distance"] for paper in paper_data])
    embeddings = np.array([paper["entity"]["embedding"] for paper in paper_data])

    # Start with the most similar paper
    selected_indices = [np.argmax(distances)]

    # Iteratively select papers
    for _ in range(1, k):
        max_combined_score = -np.inf
        next_index = -1

        for i in range(len(paper_data)):
            if i in selected_indices:
                continue

            # Compute the diversity score: minimum distance to selected papers
            diversity_score = min(
                cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                for j in selected_indices
            )

            # Combined score: similarity to input paper and diversity
            similarity_to_input = distances[i]
            combined_score = (
                similarity_to_input * 0.5 + (1 - diversity_score) * 0.5
            )  # Adjust weights

            if combined_score > max_combined_score:
                max_combined_score = combined_score
                next_index = i

        selected_indices.append(next_index)

    # Return the IDs of the selected papers
    return [paper_data[i]["id"] for i in selected_indices]

def select_diverse_papers_with_weighted_similarity(paper_data, k, diversity_weight=0.5):
    """
    Selects `k` papers that balance diversity and similarity to the input paper based on the `diversity_weight`.
    
    Args:
        paper_data (list): List of dictionaries, where each dictionary represents a paper and contains:
                           - "id": Paper ID
                           - "distance": Similarity to the input paper
                           - "embedding": Embedding of the paper
        k (int): Number of papers to select.
        diversity_weight (float): Weight between similarity and diversity (0 to 1). 
                                  1 means prioritizing diversity, 0 means prioritizing similarity.
    
    Returns:
        list: IDs of the selected papers.
    """
    # Extract distances and embeddings
    distances = np.array([paper["distance"] for paper in paper_data])
    embeddings = np.array([paper["embedding"] for paper in paper_data])
    
    # Start with the most similar paper
    selected_indices = [np.argmax(distances)]
    
    # Iteratively select papers
    for _ in range(1, k):
        max_combined_score = -np.inf
        next_index = -1
        
        for i in range(len(paper_data)):
            if i in selected_indices:
                continue
            
            # Compute the diversity score: minimum distance to selected papers
            diversity_score = min(
                cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                for j in selected_indices
            )
            
            # Combined score: similarity to input paper and diversity
            similarity_to_input = distances[i]
            combined_score = (1 - diversity_weight) * similarity_to_input + diversity_weight * (1 - diversity_score)
            
            if combined_score > max_combined_score:
                max_combined_score = combined_score
                next_index = i
        
        selected_indices.append(next_index)
    
    # Return the IDs of the selected papers
    return [paper_data[i]["id"] for i in selected_indices]
