#!/usr/bin/env python3
"""
Script to generate related works sections for papers in papers.csv using both
SimpleWorkflow and SimpleRAGWorkflow approaches with GPT-4o via Azure OpenAI.
"""

import os
import re
import time
import pandas as pd
import json
from typing import Dict, Any, List
import argparse

# Import workflow classes
from simple_agentic_workflow import SimpleWorkflow
from simple_rag_workflow import SimpleRAGWorkflow

# Import utilities for LLM clients
from citegeist.utils.llm_clients import AzureClient
from citegeist.utils.helpers import load_api_key


class RelatedWorksGenerator:
    """Generate related works sections using different baseline approaches."""

    def __init__(self, config_file: str, milvus_db_path: str = "./database.db"):
        """
        Initialize the generator.
        
        Args:
            config_file: Path to configuration file with Azure OpenAI credentials
            milvus_db_path: Path to Milvus database for SimpleRAGWorkflow
        """
        self.config_file = config_file
        self.milvus_db_path = milvus_db_path
        
        # Load API key from config file
        self.api_key = load_api_key(config_file)
        
        # Azure configuration based on workflow_demo.py
        self.azure_endpoint = "https://cai-project.openai.azure.com"
        self.azure_deployment_id = "gpt-4o"
        self.azure_api_version = "2024-02-15-preview"
        
        # Store these as None initially and initialize them only when needed
        self.azure_client = None
        self.agentic_workflow = None
        self.rag_workflow = None
    
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
    
    def _init_agentic_workflow(self):
        """Initialize the SimpleWorkflow if it hasn't been initialized yet."""
        if self.agentic_workflow is None:
            self._init_azure_client()
            self.agentic_workflow = SimpleWorkflow(self.azure_client)
        return self.agentic_workflow
    
    def _init_rag_workflow(self):
        """Initialize the SimpleRAGWorkflow if it hasn't been initialized yet."""
        if self.rag_workflow is None:
            self.rag_workflow = SimpleRAGWorkflow(
                llm_provider="azure",
                database_uri=self.milvus_db_path,
                api_key=self.api_key,
                endpoint=self.azure_endpoint,
                deployment_id=self.azure_deployment_id,
                api_version=self.azure_api_version
            )
        return self.rag_workflow
    
    def generate_with_agentic_workflow(self, abstract: str, arxiv_id: str = None, title: str = None) -> Dict[str, Any]:
        """
        Generate related works section using SimpleWorkflow.
        
        Args:
            abstract: Abstract of the paper
            arxiv_id: ArXiv ID of the paper (to exclude it from search results)
            title: Title of the paper (to exclude it from search results)
            
        Returns:
            Dictionary with related_works and other metadata
        """
        print(f"Generating related works with SimpleWorkflow...")
        workflow = self._init_agentic_workflow()
        
        # If paper has an arXiv ID or title, we need to handle it specially
        if (arxiv_id and not pd.isna(arxiv_id) and arxiv_id.strip()) or (title and title.strip()):
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
                }
            
            # Step 3: Present the search results to the user for manual filtering
            print("\nSearch results:")
            for i, paper in enumerate(search_results):
                print(f"[{i}] {paper['title']} by {', '.join(paper['authors'][:2])}")
            
            print("\nOptions:")
            print("Enter the number of a paper to remove it from the list")
            print("Enter 'C' to continue with the current list")
            
            filtered_results = list(search_results)  # Make a copy
            
            while True:
                user_input = input("Your choice: ").strip()
                
                if user_input.lower() == 'c':
                    print("Continuing with current list of papers.")
                    break
                
                try:
                    idx = int(user_input)
                    if 0 <= idx < len(filtered_results):
                        removed_paper = filtered_results.pop(idx)
                        print(f"Removed paper: {removed_paper['title']}")
                        
                        # Display the updated list
                        print("\nUpdated list:")
                        for i, paper in enumerate(filtered_results):
                            print(f"[{i}] {paper['title']} by {', '.join(paper['authors'][:2])}")
                        
                        print("\nOptions:")
                        print("Enter the number of a paper to remove it from the list")
                        print("Enter 'C' to continue with the current list")
                    else:
                        print(f"Invalid index. Please enter a number between 0 and {len(filtered_results) - 1}.")
                except ValueError:
                    print("Invalid input. Please enter a number or 'C'.")
            
            # If we have no papers left after filtering, give a message
            if not filtered_results:
                return {
                    "related_works": "No papers remaining after filtering.",
                    "search_query": search_query,
                    "papers": [],
                }
            
            # Limit to 10 papers to avoid overwhelming the LLM
            filtered_results = filtered_results[:10]
            
            # Step 5: Ask the LLM to select papers from filtered results
            selection_prompt = f"""
            I searched for papers related to your research with the query: "{search_query}"
            
            Here are the papers I found:
            """
            
            for i, paper in enumerate(filtered_results):
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
                        if 0 <= idx < len(filtered_results):
                            indices.append(idx)
                    except ValueError:
                        pass
            
            # If that fails, fall back to the original method
            if not indices:
                indices = workflow.extract_selected_indices(selection_response, len(filtered_results) - 1)
            
            # If still no valid indices, just select the first few papers
            if not indices:
                indices = list(range(min(5, len(filtered_results))))
            
            selected_indices = list(set(indices))  # Remove duplicates
            
            # Step 6: Get details for selected papers
            selected_papers = []
            for idx in selected_indices:
                if 0 <= idx < len(filtered_results):
                    paper_id = filtered_results[idx]["id"]
                    print(f"Getting details for paper {idx}: {filtered_results[idx]['title']}")
                    paper_details = workflow.get_paper_details(paper_id)
                    selected_papers.append(paper_details)
            
            if not selected_papers:
                return {
                    "related_works": "Failed to select any papers after filtering.",
                    "search_query": search_query,
                    "papers": [],
                }
            
            # Step 7: Ask the LLM to generate a related work section with only the filtered papers
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
            
            return {
                "related_works": related_works,
                "search_query": search_query,
                "papers": selected_papers,
            }
        
        # If no arXiv ID or title, just run the workflow normally
        return workflow.run(abstract)
    
    def generate_with_rag_workflow(self, abstract: str, arxiv_id: str = None, num_papers: int = 8) -> Dict[str, Any]:
        """
        Generate related works section using SimpleRAGWorkflow.
        
        Args:
            abstract: Abstract of the paper
            arxiv_id: ArXiv ID of the paper (to exclude it from search results)
            num_papers: Number of papers to retrieve
            
        Returns:
            Dictionary with related_works and citations
        """
        print(f"Generating related works with SimpleRAGWorkflow...")
        workflow = self._init_rag_workflow()
        
        # Get initial results
        result = workflow.generate_related_work(abstract, num_papers=num_papers)
        
        # If paper has an arXiv ID, we should try to filter it from citations
        if arxiv_id and not pd.isna(arxiv_id) and arxiv_id.strip():
            print(f"Checking citations for arXiv ID: {arxiv_id}")
            
            # Check if the paper is in the citations
            filtered_citations = []
            has_own_paper = False
            
            for citation in result.get('citations', []):
                if arxiv_id not in citation:
                    filtered_citations.append(citation)
                else:
                    has_own_paper = True
                    print(f"Removed citation with arXiv ID: {arxiv_id}")
            
            # If found and filtered, update the result
            if has_own_paper:
                print(f"Found paper in citations. Need to regenerate related works without it.")
                
                # This is a simplified approach - ideally we'd have access to the
                # original papers and could regenerate completely, but without that
                # we'll try to remove references to the filtered paper from the text.
                
                # Find the citation pattern in the text (e.g., "Smith et al. (2023)")
                # and attempt to remove sentences referencing it
                
                # For now, we'll just update the citations list
                result['citations'] = filtered_citations
                result['filtered_arxiv_id'] = arxiv_id
                
                # Note: Ideally, we would regenerate the related works section text
                # completely, but this would require more complex integration with
                # the SimpleRAGWorkflow class internals.
                
                # Since we only have access to the final output, we'll keep the 
                # related_works text as is but note that citations were filtered
                print("Note: Citations filtered but related_works text may still contain references to the filtered paper.")
            else:
                print(f"No matching paper found in citations. No filtering needed.")
        
        return result
    
    def process_papers(self, papers_df: pd.DataFrame, output_dir: str = "./outputs", 
                     run_agentic: bool = True, run_rag: bool = True):
        """
        Process all papers in the DataFrame and generate related works sections.
        
        Args:
            papers_df: DataFrame containing paper information
            output_dir: Directory to save output files
            run_agentic: Whether to run the SimpleWorkflow (agentic) approach
            run_rag: Whether to run the SimpleRAGWorkflow (RAG) approach
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize result DataFrames
        agentic_results = []
        rag_results = []
        
        # Process each paper
        for i, row in papers_df.iterrows():
            title = row['title']
            abstract = row['abstract']
            arxiv_id = row.get('arxiv_id', None)
            
            print(f"\nProcessing paper {i+1}/{len(papers_df)}: {title}")
            
            # Note whether the paper is already on arXiv (for logging purposes)
            if arxiv_id and not pd.isna(arxiv_id) and arxiv_id.strip():
                print(f"Paper has arXiv ID: {arxiv_id} - will exclude this paper from the retrieval results")
            
            try:
                # Generate with SimpleWorkflow if enabled
                if run_agentic:
                    agentic_result = self.generate_with_agentic_workflow(abstract, arxiv_id, title)
                    
                    # Add to results
                    agentic_results.append({
                        'title': title,
                        'arxiv_id': arxiv_id if arxiv_id and not pd.isna(arxiv_id) else "",
                        'related_works': agentic_result['related_works'],
                        'search_query': agentic_result.get('search_query', ''),
                        'papers_used': [p.get('citation', '') for p in agentic_result.get('papers', [])]
                    })
                
                # Generate with SimpleRAGWorkflow if enabled
                if run_rag:
                    rag_result = self.generate_with_rag_workflow(abstract, arxiv_id)
                    
                    # Add to results
                    rag_results.append({
                        'title': title,
                        'arxiv_id': arxiv_id if arxiv_id and not pd.isna(arxiv_id) else "",
                        'related_works': rag_result['related_works'],
                        'citations': rag_result.get('citations', [])
                    })
                
                # Add a delay to avoid rate limiting
                time.sleep(3)
            
            except Exception as e:
                print(f"Error processing paper: {title}")
                print(f"Error details: {str(e)}")
                continue
        
        # Save results to CSV files
        if run_agentic and agentic_results:
            pd.DataFrame(agentic_results).to_csv(os.path.join(output_dir, 'agentic_workflow_results.csv'), index=False)
            print(f"Agentic workflow results saved to {os.path.join(output_dir, 'agentic_workflow_results.csv')}")
            
        if run_rag and rag_results:
            pd.DataFrame(rag_results).to_csv(os.path.join(output_dir, 'rag_workflow_results.csv'), index=False)
            print(f"RAG workflow results saved to {os.path.join(output_dir, 'rag_workflow_results.csv')}")
            
        print(f"\nResults saved to {output_dir}")


def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate related works sections for papers')
    parser.add_argument('--input', type=str, default='out/papers.csv', help='Path to papers CSV file')
    parser.add_argument('--output', type=str, default='./outputs', help='Output directory for results')
    parser.add_argument('--config', type=str, default='carl_config.json', help='Path to API key configuration file')
    parser.add_argument('--database', type=str, default='./database.db', help='Path to Milvus database')
    
    # Add workflow selection arguments
    workflow_group = parser.add_mutually_exclusive_group()
    workflow_group.add_argument('--agentic-only', action='store_true', help='Run only the SimpleWorkflow (agentic) approach')
    workflow_group.add_argument('--rag-only', action='store_true', help='Run only the SimpleRAGWorkflow (RAG) approach')
    
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Check if papers CSV file exists
    if not os.path.exists(args.input):
        print(f"Error: Papers CSV file not found at {args.input}")
        return
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found at {args.config}")
        return
    
    # Load papers data
    papers_df = pd.read_csv(args.input)
    print(f"Loaded {len(papers_df)} papers from {args.input}")
    
    # Determine which workflows to run
    run_agentic = not args.rag_only  # Run agentic unless --rag-only is specified
    run_rag = not args.agentic_only  # Run RAG unless --agentic-only is specified
    
    # Display workflow selection
    if run_agentic and run_rag:
        print("Running both SimpleWorkflow and SimpleRAGWorkflow approaches")
    elif run_agentic:
        print("Running only SimpleWorkflow (agentic) approach")
    elif run_rag:
        print("Running only SimpleRAGWorkflow (RAG) approach")
    
    # Initialize generator
    generator = RelatedWorksGenerator(
        config_file=args.config,
        milvus_db_path=args.database
    )
    
    # Process papers with specified workflows
    generator.process_papers(
        papers_df, 
        output_dir=args.output,
        run_agentic=run_agentic,
        run_rag=run_rag
    )


if __name__ == "__main__":
    main()