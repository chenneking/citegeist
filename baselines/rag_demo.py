"""
Demo of the simple RAG workflow for related work generation.
This script shows how to use the SimpleRAGWorkflow class.
"""

import os
import time

from simple_rag_workflow import SimpleRAGWorkflow


def main():
    print("=== SIMPLE RAG WORKFLOW DEMO ===")
    print("This demo shows how to generate a related work section using SimpleRAGWorkflow")
    print()
    
    # Initialize the workflow with Gemini using the API key from conda environment
    workflow = SimpleRAGWorkflow(
        llm_provider="gemini",
        database_uri="./database.db",  # Path to your Milvus database
        api_key=os.environ.get("GEMINI_API_KEY"),  # This will get the key from the conda environment
        model_name="gemini-2.0-flash"
    )
    
    # Define input abstract
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
    
    print("Input Abstract:")
    print(abstract)
    print()
    
    # Find relevant papers
    print("Step 1: Finding relevant papers...")
    start_time = time.time()
    papers = workflow.find_relevant_papers(abstract, num_papers=8)
    print(f"Found {len(papers)} relevant papers in {time.time() - start_time:.2f} seconds")
    
    # Print paper details
    print("\nRelevant Papers:")
    for i, paper in enumerate(papers):
        print(f"{i+1}. {paper['id']} - Similarity: {paper['similarity']:.4f}")
        print(f"   Title (from citation): {paper['citation'].split('.')[0]}")
    print()
    
    # Generate related work section
    print("Step 2: Generating related work section...")
    start_time = time.time()
    result = workflow.generate_related_work(abstract, num_papers=len(papers))
    print(f"Generated related work section in {time.time() - start_time:.2f} seconds")
    
    # Display results
    print("\n=== GENERATED RELATED WORK SECTION ===")
    print(result["related_works"])
    
    print("\n=== PAPERS CITED ===")
    for citation in result["citations"]:
        print(f"- {citation}")


if __name__ == "__main__":
    main()