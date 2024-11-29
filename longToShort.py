import json
from utils.citations import process_arxiv_paper_with_embeddings
from bertopic import BERTopic
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def read_json_file(file_path):
    """
    Reads a JSON file from the given path.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict or list: Parsed JSON data (dictionary or list depending on the JSON structure).
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    return None

def extract_most_relevant_pages(paper_embeddings, input_string, topic_model, top_k=5):
    """
    Extracts the most relevant pages based on cosine similarity with an input string.
    
    Args:
        paper_embeddings (list): A list of papers, each containing dictionaries with "text" and "embedding".
        input_string (str): The string to compare against.
        topic_model (BERTopic): The BERTopic model for generating embeddings.
        top_k (int): The number of most relevant pages to return.

    Returns:
        list: A list of dictionaries with "paper_id", "page_number", "text", and "similarity".
    """
    # Encode the input string to get its embedding
    embedding_model = topic_model.embedding_model
    input_embedding = embedding_model.embedding_model.encode(input_string)

    results = []

    for paper_idx, paper in enumerate(paper_embeddings):
        paper_id = paper_idx  # Replace with actual paper ID if available in `paper_embeddings`
        for page_number, page_data in enumerate(paper):
            page_text = page_data["text"]
            page_embedding = page_data["embedding"]
            
            # Compute cosine similarity
            similarity = cosine_similarity(
                [input_embedding], [page_embedding]
            )[0][0]
            
            # Append results
            results.append({
                "paper_id": paper_id,  # Or use a specific ID if available
                "page_number": page_number + 1,  # Pages are 1-indexed
                "text": page_text,
                "similarity": similarity
            })

    # Sort results by similarity score in descending order
    results = sorted(results, key=lambda x: x["similarity"], reverse=True)
    
    # Return the top-k most relevant pages
    return results[:top_k]

# Example Usage
file_path = "sample_data/attention-paper.json"  # Replace with the actual file path
json_data = read_json_file(file_path)

if json_data:
    topic_model = BERTopic.load("MaartenGr/BERTopic_ArXiv")  # Load BERTopic model

    paper_embeddings = []
    for paper in json_data[0][:10]:  # Process only the first 10 papers
        arxiv_id = paper['id']  # Replace with the actual paper ID key in your JSON
        
        print(f"Processing paper: {arxiv_id}")
        result = process_arxiv_paper_with_embeddings(arxiv_id, topic_model)
        
        if result:
            paper_embeddings.append(result)
            print(f"Paper {arxiv_id}: Processed successfully.")
        else:
            print(f"Paper {arxiv_id}: No content remains after filtering.")

    # Print an example: First page text and embedding of the first processed paper
    if paper_embeddings:
        print("First paper, first page text:", paper_embeddings[0][0]["text"])
        print("First paper, first page embedding:", paper_embeddings[0][0]["embedding"])
        
# Example Usage
input_string = """The dominant sequence transduction models are based on complex recurrent or
convolutional neural networks that include an encoder and a decoder. The best
performing models also connect the encoder and decoder through an attention
mechanism. We propose a new simple network architecture, the Transformer,
based solely on attention mechanisms, dispensing with recurrence and convolutions
entirely. Experiments on two machine translation tasks show these models to
be superior in quality while being more parallelizable and requiring significantly
less time to train. Our model achieves 28.4 BLEU on the WMT 2014 Englishto-German translation task, improving over the existing best results, including
ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task,
our model establishes a new single-model state-of-the-art BLEU score of 41.8 after
training for 3.5 days on eight GPUs, a small fraction of the training costs of the
best models from the literature. We show that the Transformer generalizes well to
other tasks by applying it successfully to English constituency parsing both with
large and limited training data."""  # Input query
top_k = 20  # Number of most relevant pages to retrieve

# Assuming `paper_embeddings` contains the processed papers with embeddings
relevant_pages = extract_most_relevant_pages(paper_embeddings, input_string, topic_model, top_k)

# Print the top-k relevant pages
for result in relevant_pages:
    print(f"Paper ID: {result['paper_id']}, Page Number: {result['page_number']}")
    print(f"Similarity: {result['similarity']:.4f}")
    print(f"Text: {result['text'][:200]}...")  # Print the first 200 characters
    print("-" * 80)
        
#%%

def select_diverse_papers_with_precomputed_distances(paper_data, k):
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
    embeddings = np.array([paper['entity']["embedding"] for paper in paper_data])
    
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
            combined_score = similarity_to_input * 0.5 + (1 - diversity_score) * 0.5  # Adjust weights
            
            if combined_score > max_combined_score:
                max_combined_score = combined_score
                next_index = i
        
        selected_indices.append(next_index)
    
    # Return the IDs of the selected papers
    return [paper_data[i]["id"] for i in selected_indices]

k = 2  # Number of papers to select
selected_paper_ids = select_diverse_papers_with_precomputed_distances(json_data[0], k)

print("Selected paper IDs:", selected_paper_ids)
