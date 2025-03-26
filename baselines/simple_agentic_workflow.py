"""
A simple workflow to generate related work sections using an LLM with arXiv search.
"""

import os
import re
import arxiv
import time
from typing import List, Dict, Any, Optional, Tuple

# Use your preferred LLM client
from citegeist.utils.llm_clients import (
    LLMClient,
    GeminiClient,
    OpenAIClient,
)
from citegeist.utils.llm_clients.anthropic_client import AnthropicClient


class SimpleWorkflow:
    """
    A simple workflow that guides an LLM through searching arXiv and generating related work.
    """

    def __init__(self, llm_client: LLMClient):
        """
        Initialize the SimpleWorkflow.

        Args:
            llm_client: An LLM client
        """
        self.llm_client = llm_client
        self.arxiv_client = arxiv.Client()
    
    def search_arxiv(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search arXiv with a query and return results.

        Args:
            query: Search query
            max_results: Maximum number of results to return

        Returns:
            List of paper information
        """
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        
        results = []
        for paper in self.arxiv_client.results(search):
            results.append({
                "id": paper.entry_id.split("/")[-1],
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
            })
            time.sleep(0.2)  # Be nice to the API
        
        return results
    
    def get_paper_details(self, paper_id: str) -> Dict[str, Any]:
        """
        Get details about a paper.

        Args:
            paper_id: arXiv paper ID

        Returns:
            Paper details
        """
        search = arxiv.Search(id_list=[paper_id])
        paper = next(self.arxiv_client.results(search), None)
        
        if not paper:
            return {"error": f"No paper found with ID: {paper_id}"}
        
        # Generate citation
        authors = ", ".join(author.name for author in paper.authors)
        title = paper.title
        year = paper.published.year
        citation = f"{authors} ({year}). {title}. arXiv:{paper_id}. https://arxiv.org/abs/{paper_id}"
        
        return {
            "id": paper_id,
            "title": paper.title,
            "authors": [author.name for author in paper.authors],
            "abstract": paper.summary,
            "citation": citation,
        }
    
    def extract_search_query(self, response: str) -> Tuple[bool, str]:
        """
        Extract search query from LLM response.

        Args:
            response: LLM response

        Returns:
            (success, query)
        """
        # Look for a search call format: search("query")
        match = re.search(r'search\s*\(\s*["\']([^"\']+)["\']', response)
        if match:
            return True, match.group(1)
        
        # Look for a search call format: search("query", max_results=10)
        match = re.search(r'search\s*\(\s*["\']([^"\']+)["\']', response)
        if match:
            return True, match.group(1)
        
        # Look for an explicit query indication
        match = re.search(r'query:\s*["\']([^"\']+)["\']', response, re.IGNORECASE)
        if match:
            return True, match.group(1)
        
        # Try to extract any quoted text that might be a query
        match = re.search(r'["\']([^"\']+)["\']', response)
        if match:
            return True, match.group(1)
        
        return False, "Failed to extract search query"
    
    def extract_selected_indices(self, response: str, max_index: int) -> List[int]:
        """
        Extract selected paper indices from LLM response.

        Args:
            response: LLM response
            max_index: Maximum valid index

        Returns:
            List of selected indices
        """
        # Print the response for debugging
        print("Response to extract indices from:", response)
        
        # Strategy 1: Find the longest sequence of numbers separated by commas
        # This pattern looks for digit(s) followed by comma+spaces, allowing for any prefix
        comma_sequences = re.findall(r'[^0-9]*?(\d+(?:\s*,\s*\d+)+)', response)
        
        if comma_sequences:
            # Find the longest sequence
            longest_sequence = max(comma_sequences, key=len)
            print(f"Found comma sequence: {longest_sequence}")
            
            # Extract all numbers from the sequence
            try:
                numbers = [int(num.strip()) for num in re.findall(r'\d+', longest_sequence)]
                valid_indices = [n for n in numbers if 0 <= n <= max_index]
                if valid_indices:
                    print(f"Extracted valid indices: {valid_indices}")
                    return valid_indices
            except ValueError:
                pass
        
        # Strategy 2: If no comma sequences found, try to find any sequence with "indices are" or similar
        index_marker_patterns = [
            r"(?:indices|papers)(?:\s+are)?(?:\s*:)?\s*[^0-9]*?(\d+(?:[^0-9]+\d+)+)",
            r"selected[^0-9]*?(\d+(?:[^0-9]+\d+)+)",
        ]
        
        for pattern in index_marker_patterns:
            sequences = re.findall(pattern, response, re.IGNORECASE)
            if sequences:
                try:
                    # Extract all numbers
                    numbers = [int(num) for num in re.findall(r'\d+', sequences[0])]
                    valid_indices = [n for n in numbers if 0 <= n <= max_index]
                    if valid_indices:
                        print(f"Extracted from marker pattern: {valid_indices}")
                        return valid_indices
                except ValueError:
                    continue
        
        # Strategy 3: Last resort, extract all numbers in the response
        try:
            all_numbers = [int(num) for num in re.findall(r'\d+', response)]
            valid_indices = [n for n in all_numbers if 0 <= n <= max_index]
            if valid_indices:
                print(f"Extracted all numbers as fallback: {valid_indices}")
                return valid_indices
        except ValueError:
            pass
        
        # If nothing works, default to first few papers
        default_indices = list(range(min(5, max_index + 1)))
        print(f"No valid indices found, using defaults: {default_indices}")
        return default_indices
    
    def run(self, source_abstract: str) -> Dict[str, Any]:
        """
        Run the complete workflow.

        Args:
            source_abstract: Source paper abstract

        Returns:
            Results including related work section
        """
        # Step 1: Ask the LLM to formulate a search query
        search_prompt = f"""
        I need your help in generating a related work section for a research paper.
        
        Here's the abstract of my paper:
        "{source_abstract}"
        
        To find relevant papers, you can use the search function:
        search("your query here", max_results=10)
        
        Please search for papers related to my research by calling the search function with an appropriate query.
        You must explicitly call the search function with your query.
        """
        
        search_response = self.llm_client.get_completion(search_prompt)
        success, search_query = self.extract_search_query(search_response)
        
        if not success:
            # Fallback - create a simple query from the abstract
            words = source_abstract.split()
            search_query = " ".join([word for word in words if len(word) > 5][:10])
        
        print(f"Search query: {search_query}")
        
        # Step 2: Perform the search with the extracted query
        search_results = self.search_arxiv(search_query)
        
        if not search_results:
            return {
                "related_works": "No relevant papers found.",
                "search_query": search_query,
                "papers": [],
            }
        
        # Step 3: Ask the LLM to select papers
        selection_prompt = f"""
        I searched for papers related to your research with the query: "{search_query}"
        
        Here are the papers I found:
        """
        
        for i, paper in enumerate(search_results):
            selection_prompt += f"[{i}] {paper['title']} by {', '.join(paper['authors'][:2])}\n"
        
        selection_prompt += """
        Please select the papers that are most relevant to your research by providing their indices (e.g., 0, 1, 3).
        """
        
        selection_response = self.llm_client.get_completion(selection_prompt)
        selected_indices = self.extract_selected_indices(selection_response, len(search_results) - 1)
        
        if not selected_indices:
            selected_indices = list(range(min(5, len(search_results))))
        
        # Step 4: Get details for selected papers
        selected_papers = []
        for idx in selected_indices:
            if 0 <= idx < len(search_results):
                paper_id = search_results[idx]["id"]
                paper_details = self.get_paper_details(paper_id)
                selected_papers.append(paper_details)
        
        if not selected_papers:
            return {
                "related_works": "Failed to select any papers.",
                "search_query": search_query,
                "papers": [],
            }
        
        # Step 5: Ask the LLM to generate a related work section
        generation_prompt = f"""
        Please write a related work section for my research paper based on the following information.
        
        My paper's abstract:
        "{source_abstract}"
        
        Here are the selected relevant papers:
        """
        
        for i, paper in enumerate(selected_papers):
            generation_prompt += f"""
            Paper {i+1}: {paper['title']}
            Authors: {', '.join(paper['authors'])}
            Abstract: {paper['abstract']}
            Citation: {paper['citation']}
            
            """
        
        generation_prompt += """
        Please write a cohesive, well-structured related work section that:
        1. Groups papers with similar themes
        2. Draws connections between these papers and my research
        3. Properly cites each paper when discussing it
        4. Includes a final paragraph contextualizing my work
        
        The related work section should be 3-5 paragraphs long.
        """
        
        related_works = self.llm_client.get_completion(generation_prompt)
        
        return {
            "related_works": related_works,
            "search_query": search_query,
            "papers": selected_papers,
        }


# Example usage
def main():
    # Initialize an LLM client
    llm_client = GeminiClient(
        api_key=os.getenv("GEMINI_API_KEY"),
        model_name="gemini-2.0-flash",
    )
    
    # Create the workflow
    workflow = SimpleWorkflow(llm_client)
    
    # Example abstract
    abstract = (
        """Traditional methods for aligning Large Language Models (LLMs), such as Reinforcement Learning from 
        Human Feedback (RLHF) and Direct Preference Optimization (DPO), rely on implicit principles, 
        limiting interpretability. Constitutional AI (CAI) offers an explicit, rule-based framework 
        for guiding LLM alignment. Building on this, we refine the Inverse Constitutional AI (ICAI)
        algorithm, which extracts constitutions from preference datasets. By improving principle 
        generation, clustering, and embedding processes, our approach enhances the accuracy and 
        generalizability of extracted principles across synthetic and real-world datasets. Our results 
        highlight the potential of these principles to foster more transparent and adaptable alignment 
        methods, offering a promising direction for future advancements beyond traditional fine-tuning."""
    )
    
    # Run the workflow
    result = workflow.run(abstract)
    
    # Display the results
    print("\n=== RELATED WORK SECTION ===")
    print(result["related_works"])
    
    print("\n=== PAPERS USED ===")
    for i, paper in enumerate(result["papers"]):
        print(f"{i+1}. {paper['title']}")
        print(f"   Citation: {paper['citation']}")


if __name__ == "__main__":
    main()