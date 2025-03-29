"""
Demo of the simple workflow for related work generation using Azure OpenAI's GPT-4o.
"""

import json
from citegeist.utils.llm_clients import AzureClient
from simple_agentic_workflow import SimpleWorkflow
from citegeist.utils.helpers import load_api_key

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
    workflow = SimpleWorkflow(llm_client)
    
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
    
    print("=== SIMPLE WORKFLOW DEMO WITH AZURE GPT-4o ===")
    print("This demo shows how an LLM interacts with the arXiv search API")
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
    
    # Step 3: Ask the LLM to select papers
    print("STEP 3: Asking LLM to select relevant papers")
    selection_prompt = f"""
    I searched for papers related to your research with the query: "{search_query}"
    
    Here are the papers I found:
    """
    
    for i, paper in enumerate(search_results):
        selection_prompt += f"[{i}] {paper['title']} by {', '.join(paper['authors'][:2])}\n"
    
    selection_prompt += """
    Please select the papers that are most relevant to your research by providing their indices (e.g., 0, 1, 3).
    """
    
    print("Prompt to LLM:")
    print(selection_prompt)
    print()
    
    selection_response = llm_client.get_completion(selection_prompt)
    
    print("LLM Response:")
    print(selection_response)
    print()
    
    selected_indices = workflow.extract_selected_indices(selection_response, len(search_results) - 1)
    print(f"Selected Indices: {selected_indices}")
    print()
    
    # Step 4: Get details for selected papers
    print("STEP 4: Retrieving details for selected papers")
    selected_papers = []
    for idx in selected_indices:
        if 0 <= idx < len(search_results):
            paper_id = search_results[idx]["id"]
            print(f"Retrieving details for paper {idx}: {search_results[idx]['title']}")
            paper_details = workflow.get_paper_details(paper_id)
            selected_papers.append(paper_details)
    print()
    
    # Step 5: Ask the LLM to generate a related work section
    print("STEP 5: Asking LLM to generate related work section")
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
    
    print("=== GENERATED RELATED WORK SECTION ===")
    print(related_works)
    
    print("\n=== PAPERS USED ===")
    for i, paper in enumerate(selected_papers):
        print(f"{i+1}. {paper['title']}")
        print(f"   Citation: {paper['citation']}")

if __name__ == "__main__":
    main()