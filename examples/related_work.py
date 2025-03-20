"""
Example demonstrating how to use the citegeist pipeline to generate a related works section.
"""

import os

from citegeist.generator import Generator

if __name__ == "__main__":
    # Setup Generator
    generator = Generator(
        llm_provider="gemini",  # choice of: Azure (OpenAI Studio), Anthropic, Gemini, Mistral, and OpenAI
        api_key=os.environ.get("GEMINI_API_KEY"),
        model_name="gemini-2.0-flash",
        database_path="../database.db",  # Replace with the path to your local milvus database.db file
    )

    # Define inputs
    abstract: str = (
        "Traditional methods for aligning Large Language Models (LLMs), such as Reinforcement Learning from "
        "Human Feedback (RLHF) and Direct Preference Optimization (DPO), rely on implicit principles, "
        "limiting interpretability. Constitutional AI (CAI) offers an explicit, rule-based framework "
        "for guiding LLM alignment. Building on this, we refine the Inverse Constitutional AI (ICAI) "
        "algorithm, which extracts constitutions from preference datasets. By improving principle "
        "generation, clustering, and embedding processes, our approach enhances the accuracy and "
        "generalizability of extracted principles across synthetic and real-world datasets. Our results "
        "highlight the potential of these principles to foster more transparent and adaptable alignment "
        "methods, offering a promising direction for future advancements beyond traditional fine-tuning."
    )
    breadth: int = 10
    depth: int = 2
    diversity: float = 0.0

    # Get result
    result: dict = generator.generate_related_work(abstract=abstract, breadth=breadth, depth=depth, diversity=diversity)

    print(result["related_works"])
    print(result["citations"])
