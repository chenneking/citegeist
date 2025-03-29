"""
A simple workflow to generate related work sections using an LLM with arXiv search,
now with evaluation functionality.
"""

import os
import re
import arxiv
import time
import csv
from typing import List, Dict, Any, Optional, Tuple

# Use your preferred LLM client
from citegeist.utils.llm_clients import (
    LLMClient,
    GeminiClient,
    OpenAIClient,
)
from citegeist.utils.llm_clients.anthropic_client import AnthropicClient


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


class SimpleWorkflow:
    """
    A simple workflow that guides an LLM through searching arXiv and generating related work,
    with evaluation of paper relevance and final output quality.
    """

    def __init__(self, llm_client: LLMClient, results_csv_path: str = "evaluation_results.csv"):
        """
        Initialize the SimpleWorkflow.

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
                    'related_work_prompt'
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
    
    def save_evaluation_results(self, 
                               search_query: str,
                               paper_id: str,
                               paper_title: str,
                               relevance_score: float,
                               relevance_prompt: str,
                               related_work_score: float = None,
                               related_work_prompt: str = None):
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
                related_work_prompt
            ])
    
    def run(self, source_abstract: str, interactive: bool = True) -> Dict[str, Any]:
        """
        Run the complete workflow with evaluation.

        Args:
            source_abstract: Source paper abstract
            interactive: Whether to allow user filtering of papers

        Returns:
            Results including related work section and evaluation scores
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
                "paper_scores": [],
                "related_work_score": 0.0
            }
        
        # Step a: Get full details for all search results
        detailed_results = []
        for result in search_results:
            paper_id = result["id"]
            paper_details = self.get_paper_details(paper_id)
            if "error" not in paper_details:
                detailed_results.append(paper_details)
        
        # Step b: User filtering of search results BEFORE LLM selection
        if interactive:
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
                return {
                    "related_works": "No papers selected for related work section.",
                    "search_query": search_query,
                    "papers": [],
                    "paper_scores": [],
                    "related_work_score": 0.0
                }
        
        # Step 3: Now ask the LLM to select papers from the filtered results
        selection_prompt = f"""
        I searched for papers related to your research with the query: "{search_query}"
        
        Here are the papers I found:
        """
        
        for i, paper in enumerate(detailed_results):
            selection_prompt += f"[{i}] {paper['title']} by {', '.join(paper['authors'][:2])}\n"
        
        selection_prompt += """
        Please select the papers that are most relevant to your research by providing their indices (e.g., 0, 1, 3).
        """
        
        selection_response = self.llm_client.get_completion(selection_prompt)
        selected_indices = self.extract_selected_indices(selection_response, len(detailed_results) - 1)
        
        if not selected_indices:
            selected_indices = list(range(min(5, len(detailed_results))))
        
        # Step 4: Get the selected papers
        selected_papers = []
        for idx in selected_indices:
            if 0 <= idx < len(detailed_results):
                selected_papers.append(detailed_results[idx])
        
        if not selected_papers:
            return {
                "related_works": "Failed to select any papers.",
                "search_query": search_query,
                "papers": [],
                "paper_scores": [],
                "related_work_score": 0.0
            }
        
        # Step 6: Generate the related work section
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
        
        # Step 7: Evaluate the related work section
        related_work_prompt = generate_related_work_score_prompt(source_abstract, related_works)
        related_work_score = self.evaluate_related_work_section(source_abstract, related_works)
        
        # Step 8: Save related work evaluation results for each paper
        for paper in selected_papers:
            self.save_evaluation_results(
                search_query=search_query,
                paper_id=paper['id'],
                paper_title=paper['title'],
                relevance_score=paper['relevance_score'],  # Use the already calculated relevance score
                relevance_prompt="",  # Empty placeholder since we already saved this
                related_work_score=related_work_score,
                related_work_prompt=related_work_prompt,
                related_work_text=related_works  # Save the actual related works text
            )
        
        return {
            "related_works": related_works,
            "search_query": search_query,
            "papers": selected_papers,
            "paper_scores": paper_scores,
            "related_work_score": related_work_score
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
        
        # Step 6: Evaluate the related work section
        related_work_prompt = generate_related_work_score_prompt(source_abstract, related_works)
        related_work_score = self.evaluate_related_work_section(source_abstract, related_works)
        
        # Save related work evaluation results for each paper
        for paper in selected_papers:
            self.save_evaluation_results(
                search_query=search_query,
                paper_id=paper['id'],
                paper_title=paper['title'],
                relevance_score=0.0,  # Using 0 as a placeholder since we already saved individual scores
                relevance_prompt="",  # Empty placeholder
                related_work_score=related_work_score,
                related_work_prompt=related_work_prompt
            )
        
        return {
            "related_works": related_works,
            "search_query": search_query,
            "papers": selected_papers,
            "paper_scores": paper_scores,
            "related_work_score": related_work_score
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
    
    # Determine if interactive mode should be used
    use_interactive = input("Do you want to review papers interactively? (y/n): ").lower() == 'y'
    
    # Run the workflow
    result = workflow.run(abstract, interactive=use_interactive)
    
    # Display the results
    print("\n=== RELATED WORK SECTION ===")
    print(result["related_works"])
    
    print("\n=== PAPERS USED WITH RELEVANCE SCORES ===")
    for i, paper_score in enumerate(result["paper_scores"]):
        print(f"{i+1}. {paper_score['title']}")
        print(f"   Relevance Score: {paper_score['relevance_score']}/10")
    
    print(f"\n=== RELATED WORK SECTION QUALITY SCORE: {result['related_work_score']}/10 ===")


if __name__ == "__main__":
    main()