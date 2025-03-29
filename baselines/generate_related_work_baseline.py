#!/usr/bin/env python3
"""
Script to generate related works sections for papers in papers.csv using
SimpleWorkflow approach with GPT-4o via Azure OpenAI for generation,
and Google Vertex AI Gemini model for evaluation.
"""

import os
import re
import time
import csv
import pandas as pd
import json
from typing import Dict, Any, List, Tuple, Optional
import argparse
import logging
import random

# Import workflow class
from simple_agentic_workflow import SimpleWorkflow

# Import utilities for LLM clients
from citegeist.utils.llm_clients import AzureClient, GeminiClient
from citegeist.utils.helpers import load_api_key

# Google Cloud imports
import google
from google.oauth2 import service_account
import vertexai
from vertexai.preview.generative_models import GenerationConfig, GenerativeModel


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


class EnhancedSimpleWorkflowWithGemini(SimpleWorkflow):
    """
    Enhanced version of SimpleWorkflow that uses Gemini for evaluation functionality.
    """
    
    def __init__(self, gpt_client, gemini_client, results_csv_path: str = "evaluation_results.csv"):
        """
        Initialize the EnhancedSimpleWorkflowWithGemini.
        
        Args:
            gpt_client: The GPT (Azure) client for generation tasks
            gemini_client: The Gemini client for evaluation tasks
            results_csv_path: Path to save evaluation results
        """
        # Call the parent constructor with the GPT client for generation tasks
        super().__init__(gpt_client)
        
        # Store the Gemini client for evaluation tasks
        self.gemini_client = gemini_client
        self.results_csv_path = results_csv_path
        
        # Ensure the CSV file exists with headers
        self._initialize_results_csv()
    
    def _initialize_results_csv(self):
        """Initialize the CSV file with headers if it doesn't exist"""
        if not os.path.exists(self.results_csv_path):
            with open(self.results_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'title',
                    'arxiv_id',
                    'search_query', 
                    'paper_id', 
                    'paper_title', 
                    'relevance_score', 
                    'relevance_prompt',
                    'related_work_score',
                    'related_work_prompt',
                    'related_work_text',
                    'evaluator'
                ])
    
    def save_evaluation_results(self, 
                               title: str,
                               arxiv_id: str,
                               search_query: str,
                               paper_id: str,
                               paper_title: str,
                               relevance_score: float,
                               relevance_prompt: str,
                               related_work_score: float = None,
                               related_work_prompt: str = None,
                               related_work_text: str = None,
                               evaluator: str = "gemini"):
        """
        Save evaluation results to CSV.
        
        Args:
            title: Title of the source paper
            arxiv_id: ArXiv ID of the source paper
            search_query: Search query used
            paper_id: Paper ID
            paper_title: Paper title
            relevance_score: Relevance score
            relevance_prompt: Relevance evaluation prompt
            related_work_score: Related work section score (if applicable)
            related_work_prompt: Related work evaluation prompt (if applicable)
            related_work_text: The actual related work section text (if applicable)
            evaluator: Which model performed the evaluation (default: "gemini")
        """
        # Create a new evaluation CSV file with timestamp for this run if it doesn't exist
        if not hasattr(self, 'evaluation_csv_path_with_timestamp'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            base_path = os.path.splitext(self.results_csv_path)[0]
            self.evaluation_csv_path_with_timestamp = f"{base_path}_{timestamp}.csv"
            
            # Initialize the new CSV file with headers
            with open(self.evaluation_csv_path_with_timestamp, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'title',
                    'arxiv_id',
                    'search_query', 
                    'paper_id', 
                    'paper_title', 
                    'relevance_score', 
                    'relevance_prompt',
                    'related_work_score',
                    'related_work_prompt',
                    'related_work_text',
                    'evaluator'
                ])
            
            print(f"Created new evaluation CSV file: {self.evaluation_csv_path_with_timestamp}")
            
        # Append to the new timestamped CSV file
        with open(self.evaluation_csv_path_with_timestamp, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                title,
                arxiv_id if arxiv_id and not pd.isna(arxiv_id) else "",
                search_query,
                paper_id,
                paper_title,
                relevance_score,
                relevance_prompt,
                related_work_score,
                related_work_prompt,
                related_work_text,
                evaluator
            ])
    
    def evaluate_paper_relevance(self, source_abstract: str, target_abstract: str) -> float:
        """
        Evaluate the relevance of a paper to the source abstract using Gemini.
        
        Args:
            source_abstract: Source paper abstract
            target_abstract: Target paper abstract
            
        Returns:
            Relevance score (0-10)
        """
        prompt = generate_relevance_evaluation_prompt(source_abstract, target_abstract)
        
        # Use Gemini for evaluation
        response = self.gemini_client.get_completion(prompt)
        
        # Try to extract a numeric score from the response
        try:
            # Look for a number between 0 and 10, possibly with decimal point
            match = re.search(r'\b([0-9]|10)(\.\d+)?\b', response)
            if match:
                score = float(match.group(0))
                return score
            else:
                print(f"WARNING: Failed to extract score from Gemini response: {response}")
                # Fallback to GPT for evaluation if Gemini fails
                print("Falling back to GPT for evaluation...")
                gpt_response = self.llm_client.get_completion(prompt)
                match = re.search(r'\b([0-9]|10)(\.\d+)?\b', gpt_response)
                if match:
                    score = float(match.group(0))
                    return score
                else:
                    print(f"WARNING: Failed to extract score from GPT response too: {gpt_response}")
                    return 0.0
        except Exception as e:
            print(f"ERROR extracting score: {e}")
            return 0.0
    
    def evaluate_related_work_section(self, source_abstract: str, related_work: str) -> float:
        """
        Evaluate the quality of the generated related work section using Gemini.
        
        Args:
            source_abstract: Source paper abstract
            related_work: Generated related work section
            
        Returns:
            Quality score (0-10)
        """
        prompt = generate_related_work_score_prompt(source_abstract, related_work)
        
        # Use Gemini for evaluation
        response = self.gemini_client.get_completion(prompt)
        
        # Try to extract a numeric score from the response
        try:
            # Look for a number between 0 and 10, possibly with decimal point
            match = re.search(r'\b([0-9]|10)(\.\d+)?\b', response)
            if match:
                score = float(match.group(0))
                return score
            else:
                print(f"WARNING: Failed to extract score from Gemini response: {response}")
                # Fallback to GPT for evaluation if Gemini fails
                print("Falling back to GPT for evaluation...")
                gpt_response = self.llm_client.get_completion(prompt)
                match = re.search(r'\b([0-9]|10)(\.\d+)?\b', gpt_response)
                if match:
                    score = float(match.group(0))
                    return score
                else:
                    print(f"WARNING: Failed to extract score from GPT response too: {gpt_response}")
                    return 0.0
        except Exception as e:
            print(f"ERROR extracting score: {e}")
            return 0.0


class GeminiClient:
    """
    Client for interacting with Google's Gemini model via Vertex AI.
    """
    
    def __init__(self, project_id: str, location: str = "us-central1", credentials_path: str = None):
        """
        Initialize the Gemini client.
        
        Args:
            project_id: Google Cloud project ID
            location: Google Cloud region (default: "us-central1")
            credentials_path: Path to service account credentials JSON file
        """
        self.project_id = project_id
        self.location = location
        self.model_name = "gemini-1.5-pro" # Using the latest Gemini model
        
        # Set up credentials
        if credentials_path:
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path, 
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            # Initialize Vertex AI with credentials
            vertexai.init(project=project_id, location=location, credentials=credentials)
        else:
            # Use default credentials
            vertexai.init(project=project_id, location=location)
        
        # Initialize the model
        self.model = GenerativeModel(self.model_name)
    
    def get_completion(self, prompt: str, temperature: float = 0.0, max_output_tokens: int = 4096) -> str:
        """
        Get a completion from the Gemini model.
        
        Args:
            prompt: The prompt to send to the model
            temperature: The temperature to use for generation (default: 0.0)
            max_output_tokens: Maximum number of tokens to generate (default: 4096)
            
        Returns:
            String response from the model
        """
        try:
            generation_config = GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            return response.text
            
        except Exception as e:
            print(f"Error getting completion from Gemini: {e}")
            # Implement exponential backoff for rate limits
            if "429" in str(e) or "rate limit" in str(e).lower():
                return self._retry_with_backoff(prompt, temperature, max_output_tokens)
            raise e
    
    def _retry_with_backoff(self, prompt: str, temperature: float, max_output_tokens: int, 
                           max_retries: int = 5, initial_delay: float = 1.0) -> str:
        """
        Retry the request with exponential backoff.
        
        Args:
            prompt: The prompt to send
            temperature: The temperature to use
            max_output_tokens: Maximum number of tokens to generate
            max_retries: Maximum number of retry attempts (default: 5)
            initial_delay: Initial delay in seconds (default: 1.0)
            
        Returns:
            String response from the model
        """
        delay = initial_delay
        
        for attempt in range(max_retries):
            try:
                # Add jitter to the delay
                jittered_delay = delay * (0.5 + random.random())
                print(f"Rate limit exceeded. Retrying in {jittered_delay:.2f} seconds (attempt {attempt+1}/{max_retries})...")
                time.sleep(jittered_delay)
                
                # Try again
                generation_config = GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )
                
                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                
                return response.text
                
            except Exception as e:
                print(f"Retry failed: {e}")
                # Increase delay exponentially
                delay *= 2
                
                # If this is not a rate limit error or it's the last retry, raise the exception
                if ("429" not in str(e) and "rate limit" not in str(e).lower()) or attempt == max_retries - 1:
                    raise e
        
        raise Exception("Exceeded maximum retries due to rate limiting")


class EnhancedRelatedWorksGeneratorWithGemini:
    """Generate related works sections with Gemini for evaluation."""

    def __init__(self, api_key_file: str, gemini_credentials_path: str, 
                 gemini_project_id: str, evaluation_csv_path: str = "evaluation_results.csv"):
        """
        Initialize the generator.
        
        Args:
            api_key_file: Path to Azure API key file
            gemini_credentials_path: Path to Google Cloud service account credentials JSON file
            gemini_project_id: Google Cloud project ID
            evaluation_csv_path: Path to save evaluation results
        """
        self.api_key_file = api_key_file
        self.gemini_credentials_path = gemini_credentials_path
        self.gemini_project_id = gemini_project_id
        self.evaluation_csv_path = evaluation_csv_path
        
        # Load Azure API key
        with open(api_key_file, 'r') as f:
            key_data = json.load(f)
            self.api_key = key_data.get("secret_key")
        
        if not self.api_key:
            raise ValueError("Could not find 'secret_key' in the API key file.")
            
        # Azure configuration based on workflow_demo.py
        self.azure_endpoint = "https://cai-project.openai.azure.com"
        self.azure_deployment_id = "gpt-4o"
        self.azure_api_version = "2024-02-15-preview"
        
        # Store these as None initially and initialize them only when needed
        self.azure_client = None
        self.gemini_client = None
        self.enhanced_workflow = None
    
    def _init_azure_client(self):
        """Initialize the Azure OpenAI client if it hasn't been initialized yet."""
        if self.azure_client is None:
            self.azure_client = AzureClient(
                api_key=self.api_key,
                endpoint=self.azure_endpoint,
                deployment_id=self.azure_deployment_id,
                api_version=self.azure_api_version
            )
        return self.azure_client
    
    def _init_gemini_client(self):
        """Initialize the Gemini client if it hasn't been initialized yet."""
        if self.gemini_client is None:
            self.gemini_client = GeminiClient(
                project_id=self.gemini_project_id,
                location="us-central1",
                credentials_path=self.gemini_credentials_path
            )
        return self.gemini_client
    
    def _init_enhanced_workflow(self):
        """Initialize the EnhancedSimpleWorkflowWithGemini if it hasn't been initialized yet."""
        if self.enhanced_workflow is None:
            self._init_azure_client()
            self._init_gemini_client()
            self.enhanced_workflow = EnhancedSimpleWorkflowWithGemini(
                gpt_client=self.azure_client,
                gemini_client=self.gemini_client,
                results_csv_path=self.evaluation_csv_path
            )
        return self.enhanced_workflow
    
    def generate_with_enhanced_workflow(self, title: str, abstract: str, arxiv_id: str = None) -> Dict[str, Any]:
        """
        Generate related works section using GPT with Gemini for evaluation.
        
        Args:
            title: Title of the paper
            abstract: Abstract of the paper
            arxiv_id: ArXiv ID of the paper (to exclude it from search results)
            
        Returns:
            Dictionary with related_works, evaluation scores, and other metadata
        """
        print(f"Generating related works with GPT, evaluating with Gemini...")
        workflow = self._init_enhanced_workflow()
        
        # Override arXiv ID for Classical transport paper (fix for the corrupted ID)
        if title == "Classical transport in a maximally chaotic chain" and arxiv_id == "2411.19828":
            print(f"Using corrected arXiv ID for this paper")
            arxiv_id = "2411.19828"  # Using the correct ID directly
        
        # Step 1: Run the workflow to get the search query
        search_prompt = f"""
        I need your help in generating a related work section for a research paper.
        
        Here's the abstract of my paper:
        "{abstract}"
        
        To find relevant papers, you can use the search function:
        search("your query here", max_results=15)
        
        Please search for papers related to my research by calling the search function with an appropriate query.
        The query should focus on the research AREA and CONCEPTS, not specific paper titles.
        You must explicitly call the search function with your query.
        """
        
        search_response = workflow.llm_client.get_completion(search_prompt)
        success, search_query = workflow.extract_search_query(search_response)
        
        if not success:
            # Fallback - create a simple query from the abstract
            words = abstract.split()
            search_query = " ".join([word for word in words if len(word) > 5][:10])
        
        print(f"Search query: {search_query}")
        
        # Step 2: Perform the search with the extracted query
        search_results = workflow.search_arxiv(search_query, max_results=20)  # Get more results initially
        
        if not search_results:
            return {
                "related_works": "No relevant papers found.",
                "search_query": search_query,
                "papers": [],
                "paper_scores": [],
                "related_work_score": 0.0
            }
        
        # Get full details for all search results
        detailed_results = []
        print("Retrieving details for all search results...")
        for result in search_results:
            paper_id = result["id"]
            if '.' in paper_id:
                paper_details = workflow.get_paper_details(paper_id)
                if "error" not in paper_details:
                    detailed_results.append(paper_details)
        
        # Step 3: Present the search results to the user for manual filtering
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
        
        # Limit to 10 papers to avoid overwhelming the LLM
        if len(detailed_results) > 10:
            print(f"Limiting to 10 papers from {len(detailed_results)} filtered results.")
            detailed_results = detailed_results[:10]
        
        # Step 4: Ask the LLM to select papers from filtered results
        selection_prompt = f"""
        I searched for papers related to your research with the query: "{search_query}"
        
        Here are the papers I found:
        """
        
        for i, paper in enumerate(detailed_results):
            selection_prompt += f"[{i}] {paper['title']} by {', '.join(paper['authors'][:2])}\n"
        
        selection_prompt += """
        Please select the papers that are most relevant to your research by providing their indices (e.g., 0, 1, 3).
        """
        
        selection_response = workflow.llm_client.get_completion(selection_prompt)
        
        # Use a more robust extraction method
        indices = []
        # Look specifically for numbers surrounded by brackets - e.g., [0], [1], etc.
        bracket_matches = re.findall(r'\[(\d+)\]', selection_response)
        if bracket_matches:
            for match in bracket_matches:
                try:
                    idx = int(match)
                    if 0 <= idx < len(detailed_results):
                        indices.append(idx)
                except ValueError:
                    pass
        
        # If that fails, fall back to the original method
        if not indices:
            indices = workflow.extract_selected_indices(selection_response, len(detailed_results) - 1)
        
        # If still no valid indices, just select the first few papers
        if not indices:
            indices = list(range(min(5, len(detailed_results))))
        
        selected_indices = list(set(indices))  # Remove duplicates
        print(f"Selected indices: {selected_indices}")
        
        # Step 5: Get the selected papers
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
        
        # Step 6: Evaluate the relevance of the selected papers using Gemini
        print("Evaluating relevance of selected papers using Gemini...")
        paper_scores = []
        
        for paper in selected_papers:
            print(f"Evaluating relevance of: {paper['title']}")
            relevance_prompt = generate_relevance_evaluation_prompt(abstract, paper['abstract'])
            relevance_score = workflow.evaluate_paper_relevance(abstract, paper['abstract'])
            
            # Add relevance score to paper details
            paper['relevance_score'] = relevance_score
            
            paper_scores.append({
                "paper_id": paper['id'],
                "title": paper['title'],
                "relevance_score": relevance_score
            })
            
            print(f"Relevance Score from Gemini: {relevance_score}/10")
        
        # Step 7: Generate the related work section using GPT
        print("Generating related work section with GPT...")
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
        
        related_works = workflow.llm_client.get_completion(generation_prompt)
        
        # Step 8: Evaluate the related work section using Gemini
        print("Evaluating the related work section using Gemini...")
        related_work_prompt = generate_related_work_score_prompt(abstract, related_works)
        related_work_score = workflow.evaluate_related_work_section(abstract, related_works)
        print(f"Related Work Quality Score from Gemini: {related_work_score}/10")
        
        # Step 9: Save evaluation results for each paper
        print("Saving evaluation results to CSV...")
        for paper in selected_papers:
            workflow.save_evaluation_results(
                title=title,
                arxiv_id=arxiv_id if arxiv_id and not pd.isna(arxiv_id) else "",
                search_query=search_query,
                paper_id=paper['id'],
                paper_title=paper['title'],
                relevance_score=paper['relevance_score'],
                relevance_prompt=generate_relevance_evaluation_prompt(abstract, paper['abstract']),
                related_work_score=related_work_score,
                related_work_prompt=related_work_prompt,
                related_work_text=related_works,
                evaluator="gemini"
            )
        
        return {
            "related_works": related_works,
            "search_query": search_query,
            "papers": selected_papers,
            "paper_scores": paper_scores,
            "related_work_score": related_work_score
        }
    
    def process_papers(self, papers_df: pd.DataFrame, output_dir: str = "./outputs", retry_failed_only: bool = False, failed_papers: List[str] = None):
        """
        Process all papers in the DataFrame and generate related works sections with Gemini evaluation.
        
        Args:
            papers_df: DataFrame containing paper information
            output_dir: Directory to save output files
            retry_failed_only: If True, only retry papers that failed in previous runs
            failed_papers: List of paper titles that failed and should be retried
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a unique filename for this run
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if retry_failed_only:
            results_csv_path = os.path.join(output_dir, f'retry_gemini_results_{timestamp}.csv')
        else:
            results_csv_path = os.path.join(output_dir, f'gpt_gemini_workflow_results_{timestamp}.csv')
        
        print(f"Results will be saved to: {results_csv_path}")
        
        # Default failed papers list - if specified, use only these papers
        if failed_papers is None:
            failed_papers = [
                "Classical transport in a maximally chaotic chain"
            ]
        
        # Initialize result list
        enhanced_results = []
        
        # Count for tracking
        papers_to_process = 0
        papers_processed = 0
        
        # Process each paper
        for i, row in papers_df.iterrows():
            title = row['title']
            abstract = row['abstract']
            arxiv_id = row.get('arxiv_id', None)
            
            # Skip if not in the failed papers list and we're only retrying failed papers
            if retry_failed_only:
                # Check if the paper is in the failed papers list
                if not any(title.lower().strip() == failed_paper.lower().strip() or 
                         title.lower().strip() in failed_paper.lower().strip() or
                         failed_paper.lower().strip() in title.lower().strip()
                         for failed_paper in failed_papers):
                    print(f"\nSkipping paper {i+1}/{len(papers_df)}: {title} (not in the failed papers list)")
                    continue
            
            papers_to_process += 1
            print(f"\nProcessing paper {i+1}/{len(papers_df)}: {title}")
            
            # Note whether the paper is already on arXiv (for logging purposes)
            if arxiv_id and not pd.isna(arxiv_id) and arxiv_id.strip():
                print(f"Paper has arXiv ID: {arxiv_id} - will exclude this paper from the retrieval results")
            
            try:
                # Generate with GPT, evaluate with Gemini
                enhanced_result = self.generate_with_enhanced_workflow(title, abstract, arxiv_id)
                papers_processed += 1
                
                # Add to results
                new_result = {
                    'title': title,
                    'arxiv_id': arxiv_id if arxiv_id and not pd.isna(arxiv_id) else "",
                    'related_works': enhanced_result['related_works'],
                    'search_query': enhanced_result.get('search_query', ''),
                    'papers_used': [p.get('citation', '') for p in enhanced_result.get('papers', [])],
                    'paper_relevance_scores': [ps for ps in enhanced_result.get('paper_scores', [])],
                    'related_work_score': enhanced_result.get('related_work_score', 0.0),
                    'evaluator': 'gemini'
                }
                
                # Add new result
                enhanced_results.append(new_result)
                
                # Add a delay to avoid rate limiting
                time.sleep(3)
                
                # Save incremental results after each successful paper
                if enhanced_results:
                    results_df = pd.DataFrame(enhanced_results)
                    
                    # Convert the list of paper scores to a string representation for CSV storage
                    results_df['paper_relevance_scores'] = results_df['paper_relevance_scores'].apply(
                        lambda x: json.dumps(x) if isinstance(x, list) else x
                    )
                    
                    # Save to new CSV file
                    results_df.to_csv(results_csv_path, index=False)
                    print(f"Saved incremental results to {results_csv_path}")
            
            except Exception as e:
                print(f"Error processing paper: {title}")
                print(f"Error details: {str(e)}")
                continue
        
        print(f"\nProcessed {papers_processed}/{papers_to_process} papers")
        print(f"Results saved to {results_csv_path}")
        print(f"Detailed evaluation results saved to {self.evaluation_csv_path}")


def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate related works sections using GPT with Gemini evaluation')
    parser.add_argument('--input', type=str, default='../evaluation/out/papers.csv', help='Path to papers CSV file')
    parser.add_argument('--output', type=str, default='./outputs', help='Output directory for results')
    parser.add_argument('--api-key', type=str, default='../api_key.json', help='Path to Azure API key file')
    parser.add_argument('--gemini-credentials', type=str, default='../cs6158-structuralunderstanding-2647462afe3e.json', 
                        help='Path to Google Cloud service account credentials JSON file')
    parser.add_argument('--gemini-project', type=str, default='cs6158-structuralunderstanding', 
                        help='Google Cloud project ID for Gemini')
    parser.add_argument('--evaluation', type=str, default='evaluation_results.csv', help='Path to save evaluation results')
    parser.add_argument('--retry-failed', action='store_true', help='Only retry papers that failed in previous runs')
    parser.add_argument('--failed-papers', type=str, help='Comma-separated list of paper titles that failed and should be retried')
    
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Check if papers CSV file exists
    if not os.path.exists(args.input):
        print(f"Error: Papers CSV file not found at {args.input}")
        return
    
    # Check if Azure API key file exists
    if not os.path.exists(args.api_key):
        print(f"Error: Azure API key file not found at {args.api_key}")
        return
    
    # Check if Gemini credentials file exists
    if not os.path.exists(args.gemini_credentials):
        print(f"Error: Gemini credentials file not found at {args.gemini_credentials}")
        return
    
    # Load papers data
    papers_df = pd.read_csv(args.input)
    print(f"Loaded {len(papers_df)} papers from {args.input}")
    
    # Process failed papers list if provided
    failed_papers = None
    if args.failed_papers:
        failed_papers = [title.strip() for title in args.failed_papers.split(',')]
        print(f"Will retry these specific papers: {failed_papers}")
    else:
        # Default list of failed papers from the error log - use EXACT titles
        if args.retry_failed:
            failed_papers = [
                "Classical transport in a maximally chaotic chain"
            ]
            print(f"Using default list of failed papers: {failed_papers}")
    
    # Initialize generator
    generator = EnhancedRelatedWorksGeneratorWithGemini(
        api_key_file=args.api_key,
        gemini_credentials_path=args.gemini_credentials,
        gemini_project_id=args.gemini_project,
        evaluation_csv_path=args.evaluation
    )
    
    # Process papers with enhanced workflow
    generator.process_papers(
        papers_df, 
        output_dir=args.output,
        retry_failed_only=args.retry_failed,
        failed_papers=failed_papers
    )


if __name__ == "__main__":
    main()