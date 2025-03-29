"""
Demo of the enhanced workflow for related work generation with evaluation using Azure OpenAI's GPT-4o.
"""

import json
import csv
import os
import re
import arxiv
import time
from typing import List, Dict, Any, Optional, Tuple

from citegeist.utils.llm_clients import AzureClient
from citegeist.utils.helpers import load_api_key


def generate_relevance_evaluation_prompt(source_abstract: str, target_abstract: str) -> str:
    """
    Generates an evaluation prompt to utilize LLM as a judge to determine the relevance with regard to the source abstract
    :param source_abstract: Abstract of source paper
    :param target_abstract: Abstract of target paper
    :return: Prompt String
    """
    prompt = f"""
        You are given two paper abstracts: the first is the source paper abstract, and the second is a related work paper abstract. Your task is to assess the relevance of the related work abstract to the source paper abstract on a scale of 0 to 10, where:
        
        - 0 means no relevance at all (completely unrelated).
        - 10 means the highest relevance (directly related and closely aligned with the source paper's topic and content).
        
        Consider factors such as:
        - Topic alignment: Does the related work paper address a similar research problem or area as the source paper?
        - Methodology: Does the related work discuss methods or techniques similar to those in the source paper?
        - Findings or contributions: Are the findings or contributions of the related work closely related to the source paper's content or conclusions?
        - The relationship between the two papers, such as whether the related work builds on, contrasts, or expands the source paper's work.
        
        Provide a score (0–10) and a brief explanation of your reasoning for the assigned score.
        
        Source Paper Abstract:
        {source_abstract}
        
        Related Work Paper Abstract:
        {target_abstract}
        
        Please provide only the score as your reply. Do not produce any other output, including things like formatting or markdown. Only the score.
    """
    return prompt


def generate_related_work_score_prompt(source_abstract: str, related_work: str) -> str:
    """
    Generates an evaluation prompt to score the quality of the generated related work section
    :param source_abstract: Abstract of source paper
    :param related_work: Generated related work section
    :return: Prompt String
    """
    return f"""
    Source Abstract:
    {source_abstract}
    
    Related Works Section:
    {related_work}
    
    Objective:
    Evaluate this related works section with regard to the source abstract provided.
    
    Consider factors such as comprehensiveness, clarity of writing, relevance, etc. when making your decision.
    If invalid citations occur, consider the information to be invalid (or even completely false).
    
    Exclusively respond with your choice of rating. For this purpose you can assign a score from 0-10 where 0 is worst and 10 is best.
    
    - **0**: Completely irrelevant, unclear, or inaccurate. 
     *Example*: The section does not address the Source Abstract's topics and contains multiple invalid citations.
      
    - **5**: Somewhat relevant but lacks comprehensiveness, clarity or relevance.
     *Example*: The section references a few relevant works but also includes irrelevant ones and has minor errors.
      
    - **10**: Exceptionally relevant, comprehensive, clear, and accurate.
      *Example*: The section thoroughly addresses all key topics, includes all relevant works, and is clearly written with no factual errors.
    
    Do not include anything else in your output.
    """


class EnhancedWorkflow:
    """
    An enhanced workflow that guides an LLM through searching arXiv, generating related work,
    and evaluating the relevance and quality of the results.
    """

    def __init__(self, llm_client, results_csv_path: str = "evaluation_results.csv"):
        """
        Initialize the EnhancedWorkflow.

        Args:
            llm_client: An LLM client
            results_csv_path: Path to save evaluation results
        """
        self.llm_client = llm_client
        self.arxiv_client = arxiv.Client()
        self.results_csv_path = results_csv_path
        
        # Ensure the CSV file exists with headers
        self._initialize_results_csv()
    
    def _initialize_results_csv(self):
        """Initialize the CSV file with headers if it doesn't exist"""
        if not os.path.exists(self.results_csv_path):
            with open(self.results_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'search_query', 
                    'paper_id', 
                    'paper_title', 
                    'relevance_score', 
                    'relevance_prompt',
                    'related_work_score',
                    'related_work_prompt',
                    'related_work_text'
                ])
    
    def save_evaluation_results(self, 
                               search_query: str,
                               paper_id: str,
                               paper_title: str,
                               relevance_score: float,
                               relevance_prompt: str,
                               related_work_score: float = None,
                               related_work_prompt: str = None,
                               related_work_text: str = None):
        """
        Save evaluation results to CSV.
        
        Args:
            search_query: Search query used
            paper_id: Paper ID
            paper_title: Paper title
            relevance_score: Relevance score
            relevance_prompt: Relevance evaluation prompt
            related_work_score: Related work section score (if applicable)
            related_work_prompt: Related work evaluation prompt (if applicable)
            related_work_text: The actual related work section text (if applicable)
        """
        with open(self.results_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                search_query,
                paper_id,
                paper_title,
                relevance_score,
                relevance_prompt,
                related_work_score,
                related_work_prompt,
                related_work_text
            ])
    
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
    
    def evaluate_paper_relevance(self, source_abstract: str, target_abstract: str) -> float:
        """
        Evaluate the relevance of a paper to the source abstract.
        
        Args:
            source_abstract: Source paper abstract
            target_abstract: Target paper abstract
            
        Returns:
            Relevance score (0-10)
        """
        prompt = generate_relevance_evaluation_prompt(source_abstract, target_abstract)
        response = self.llm_client.get_completion(prompt)
        
        # Try to extract a numeric score from the response
        try:
            # Look for a number between 0 and 10, possibly with decimal point
            match = re.search(r'\b([0-9]|10)(\.\d+)?\b', response)
            if match:
                score = float(match.group(0))
                return score
            else:
                print(f"WARNING: Failed to extract score from response: {response}")
                return 0.0
        except Exception as e:
            print(f"ERROR extracting score: {e}")
            return 0.0
    
    def evaluate_related_work_section(self, source_abstract: str, related_work: str) -> float:
        """
        Evaluate the quality of the generated related work section.
        
        Args:
            source_abstract: Source paper abstract
            related_work: Generated related work section
            
        Returns:
            Quality score (0-10)
        """
        prompt = generate_related_work_score_prompt(source_abstract, related_work)
        response = self.llm_client.get_completion(prompt)
        
        # Try to extract a numeric score from the response
        try:
            # Look for a number between 0 and 10, possibly with decimal point
            match = re.search(r'\b([0-9]|10)(\.\d+)?\b', response)
            if match:
                score = float(match.group(0))
                return score
            else:
                print(f"WARNING: Failed to extract score from response: {response}")
                return 0.0
        except Exception as e:
            print(f"ERROR extracting score: {e}")
            return 0.0


def main():
    # Load API key from JSON file
    api_key = load_api_key("../api_key.json")
    
    # Initialize the Azure OpenAI client with GPT-4o
    llm_client = AzureClient(
        api_key=api_key,
        endpoint="https://cai-project.openai.azure.com",  # Update this if needed
        deployment_id="gpt-4o",  # Update if your deployment has a different name
        api_version="2024-02-15-preview",  # Make sure this is the correct API version
    )
    
    # Create the workflow
    workflow = EnhancedWorkflow(llm_client)
    
    # Define the source abstract
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
    
    print("=== ENHANCED WORKFLOW DEMO WITH AZURE GPT-4o ===")
    print("This demo shows how an LLM interacts with the arXiv search API with evaluation functionality")
    print()
    
    # Step 1: Ask the LLM to formulate a search query
    print("STEP 1: Asking LLM to formulate a search query")
    search_prompt = f"""
    I need your help in generating a related work section for a research paper.
    
    Here's the abstract of my paper:
    "{abstract}"
    
    To find relevant papers, you can use the search function:
    search("your query here", max_results=10)
    
    Please search for papers related to my research by calling the search function with an appropriate query.
    You must explicitly call the search function with your query.
    """
    
    print("Prompt to LLM:")
    print(search_prompt)
    print()
    
    search_response = llm_client.get_completion(search_prompt)
    
    print("LLM Response:")
    print(search_response)
    print()
    
    success, search_query = workflow.extract_search_query(search_response)
    print(f"Extracted Query: {search_query} (Success: {success})")
    print()
    
    # Step 2: Execute the search
    print("STEP 2: Executing search with extracted query")
    search_results = workflow.search_arxiv(search_query)
    
    print(f"Found {len(search_results)} papers:")
    for i, paper in enumerate(search_results):
        print(f"[{i}] {paper['title']} by {', '.join(paper['authors'][:2])}")
    print()
    
    # Step 3: User filtering of search results BEFORE LLM selection
    # Get full details for all search results
    detailed_results = []
    print("Retrieving details for all search results...")
    for result in search_results:
        paper_id = result["id"]
        paper_details = workflow.get_paper_details(paper_id)
        if "error" not in paper_details:
            detailed_results.append(paper_details)
    
    # Display the full list of search results
    print("\n=== SEARCH RESULTS ===")
    for i, paper in enumerate(detailed_results):
        print(f"[{i}] {paper['title']}")
        print(f"    Authors: {', '.join(paper['authors'])}")
        print(f"    Abstract: {paper['abstract'][:150]}...")
        print()
        
    # List of papers to remove
    papers_to_remove = []
    
    # Prompt user for filtering
    print("\nEnter the index of a paper to remove, or 'C' to continue with all papers.")
    while True:
        user_input = input("Enter index or 'C': ").upper()
        
        if user_input == 'C':
            break
        
        try:
            idx = int(user_input)
            if 0 <= idx < len(detailed_results):
                papers_to_remove.append(idx)
                print(f"Paper {idx} marked for removal. Enter another index or 'C' to continue.")
            else:
                print(f"Index {idx} is out of range. Try again.")
        except ValueError:
            print("Invalid input. Enter a number or 'C'.")
    
    # Remove papers
    if papers_to_remove:
        # Sort in reverse order to avoid index shifting when removing
        for idx in sorted(papers_to_remove, reverse=True):
            print(f"Removing: {detailed_results[idx]['title']}")
            detailed_results.pop(idx)
    
    if not detailed_results:
        print("No papers selected. Exiting.")
        return
    
    # Step 4: Ask the LLM to select papers from filtered results
    print("STEP 4: Asking LLM to select relevant papers from filtered results")
    selection_prompt = f"""
    I searched for papers related to your research with the query: "{search_query}"
    
    Here are the papers I found:
    """
    
    for i, paper in enumerate(detailed_results):
        selection_prompt += f"[{i}] {paper['title']} by {', '.join(paper['authors'][:2])}\n"
    
    selection_prompt += """
    Please select the papers that are most relevant to my research by providing their indices (e.g., 0, 1, 3).
    """
    
    print("Prompt to LLM:")
    print(selection_prompt)
    print()
    
    selection_response = llm_client.get_completion(selection_prompt)
    
    print("LLM Response:")
    print(selection_response)
    print()
    
    selected_indices = workflow.extract_selected_indices(selection_response, len(detailed_results) - 1)
    print(f"Selected Indices: {selected_indices}")
    print()
    
    # Step 5: Get the selected papers
    print("STEP 5: Using selected papers")
    selected_papers = []
    for idx in selected_indices:
        if 0 <= idx < len(detailed_results):
            selected_papers.append(detailed_results[idx])
    
    print(f"Selected {len(selected_papers)} papers for related work section")
    for i, paper in enumerate(selected_papers):
        print(f"[{i}] {paper['title']}")
    print()
    
    # Step 6: Evaluate the relevance of the selected papers
    print("STEP 6: Evaluating relevance of selected papers")
    paper_scores = []
    
    for paper in selected_papers:
        print(f"Evaluating relevance of: {paper['title']}")
        relevance_prompt = generate_relevance_evaluation_prompt(abstract, paper['abstract'])
        relevance_score = workflow.evaluate_paper_relevance(abstract, paper['abstract'])
        
        # Add relevance score to paper details
        paper['relevance_score'] = relevance_score
        
        # Save paper evaluation results
        workflow.save_evaluation_results(
            search_query=search_query,
            paper_id=paper['id'],
            paper_title=paper['title'],
            relevance_score=relevance_score,
            relevance_prompt=relevance_prompt
        )
        
        paper_scores.append({
            "paper_id": paper['id'],
            "title": paper['title'],
            "relevance_score": relevance_score
        })
        
        print(f"Relevance Score: {relevance_score}/10")
    print()
    
    # Step 7: Generate the related work section
    print("STEP 7: Generating related work section")
    generation_prompt = f"""
    Please write a related work section for my research paper based on the following information.
    
    My paper's abstract:
    "{abstract}"
    
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
    
    print("Prompt to LLM (truncated):")
    print(generation_prompt[:500] + "...")
    print()
    
    related_works = llm_client.get_completion(generation_prompt)
    
    # Step 8: Evaluate the related work section
    print("STEP 8: Evaluating the related work section")
    related_work_prompt = generate_related_work_score_prompt(abstract, related_works)
    related_work_score = workflow.evaluate_related_work_section(abstract, related_works)
    
    # Step 9: Save related work evaluation results for each paper
    print("STEP 9: Saving evaluation results to CSV")
    for paper in selected_papers:
        workflow.save_evaluation_results(
            search_query=search_query,
            paper_id=paper['id'],
            paper_title=paper['title'],
            relevance_score=paper['relevance_score'],  # Use the already calculated relevance score
            relevance_prompt="",  # Empty placeholder since we already saved this
            related_work_score=related_work_score,
            related_work_prompt=related_work_prompt,
            related_work_text=related_works  # Save the actual related works text
        )
    
    print("=== EVALUATION RESULTS ===")
    print(f"Related Work Quality Score: {related_work_score}/10")
    print()
    
    print("=== GENERATED RELATED WORK SECTION ===")
    print(related_works)
    
    print("\n=== PAPERS USED WITH RELEVANCE SCORES ===")
    for i, paper_score in enumerate(paper_scores):
        print(f"{i+1}. {paper_score['title']}")
        print(f"   Relevance Score: {paper_score['relevance_score']}/10")


if __name__ == "__main__":
    main()