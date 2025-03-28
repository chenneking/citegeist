"""
Simple RAG workflow for generating related work sections.
"""

import time
from typing import Dict, List, Union, Any

from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient

from citegeist.utils.llm_clients import create_client
from citegeist.utils.citations import get_arxiv_abstract, get_arxiv_citation, filter_citations


class SimpleRAGWorkflow:
    """A simple RAG workflow for generating related works sections from abstracts."""

    def __init__(
        self,
        llm_provider: str,
        database_uri: str,
        sentence_embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        **llm_kwargs
    ):
        """
        Initialize the SimpleRAGWorkflow.
        
        Args:
            llm_provider: LLM provider name ('azure', 'openai', 'anthropic', 'gemini')
            database_uri: Path to the Milvus database
            sentence_embedding_model_name: Name of the sentence transformer embedding model
            **llm_kwargs: Provider-specific configuration arguments for the LLM client
        """
        # Initialize models and clients
        self.embedding_model = SentenceTransformer(sentence_embedding_model_name)
        self.db_client = MilvusClient(database_uri)
        self.llm_client = create_client(llm_provider, **llm_kwargs)
        
    def find_relevant_papers(self, abstract: str, num_papers: int = 8) -> List[Dict[str, Any]]:
        """
        Find the top N most relevant papers based on embedding similarity.
        
        Args:
            abstract: The input abstract text
            num_papers: Number of papers to retrieve
            
        Returns:
            List of paper data dictionaries
        """
        # Embed the input abstract
        embedded_abstract = self.embedding_model.encode(abstract)
        
        # Query the vector database for similar papers
        query_results = self.db_client.search(
            collection_name="abstracts",
            data=[embedded_abstract],
            limit=num_papers,
            anns_field="embedding",
            search_params={"metric_type": "COSINE", "params": {}}
        )
        
        # Convert results to a usable format
        papers = []
        for result in query_results[0]:
            arxiv_id = result["id"]
            
            # Add a small delay to avoid rate limiting
            time.sleep(0.2)
            
            # Get full abstract and format citation
            try:
                arxiv_abstract = get_arxiv_abstract(arxiv_id)
                citation = get_arxiv_citation(arxiv_id)
                
                papers.append({
                    "id": arxiv_id,
                    "abstract": arxiv_abstract,
                    "citation": citation,
                    "similarity": result["distance"]
                })
            except Exception as e:
                print(f"Error retrieving details for paper {arxiv_id}: {e}")
                continue
            
        return papers
    
    def generate_prompt(self, abstract: str, papers: List[Dict[str, Any]]) -> str:
        """
        Generate the prompt for the LLM to create a related work section.
        
        Args:
            abstract: The input abstract
            papers: List of relevant papers with abstracts and citations
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""
        I need a well-written "Related Work" section for my research paper.
        
        Here's the abstract of my paper:
        "{abstract}"
        
        Here are the abstracts of relevant papers I've found:
        """
        
        for i, paper in enumerate(papers):
            prompt += f"""
            Paper {i+1}:
            Abstract: {paper['abstract']}
            Citation: {paper['citation']}
            
            """
        
        prompt += """
        Please write a cohesive, well-structured related work section that:
        1. Groups papers with similar themes
        2. Draws connections between these papers and my research
        3. Properly cites each paper when discussing it (add the citation immediately after each reference)
        4. Includes a final paragraph contextualizing my work
        
        The related work section should be 3-5 paragraphs long.
        """
        
        return prompt
    
    def generate_related_work(self, abstract: str, num_papers: int = 8) -> Dict[str, Union[str, List[str]]]:
        """
        Generate a related work section using a simple RAG approach.
        
        Args:
            abstract: The input abstract text
            num_papers: Number of papers to consider
            
        Returns:
            Dictionary with 'related_works' text and 'citations' list
        """
        # Find relevant papers
        relevant_papers = self.find_relevant_papers(abstract, num_papers)
        
        # Generate the prompt
        prompt = self.generate_prompt(abstract, relevant_papers)
        
        # Generate the related work section using the LLM
        related_works_section = self.llm_client.get_completion(prompt)
        
        # Extract citation strings that were actually used
        filtered_citations = filter_citations(
            related_works_section=related_works_section, 
            citation_strings=[paper["citation"] for paper in relevant_papers]
        )
        
        return {
            "related_works": related_works_section,
            "citations": filtered_citations
        }