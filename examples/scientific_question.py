"""
Example demonstrating how to use the citegeist pipeline to generate an answer to a scientific question.
"""

import os

from citegeist.generator import Generator

if __name__ == "__main__":
    # Setup Generator
    generator = Generator(
        llm_provider="gemini",  # choice of: Azure (OpenAI Studio), Anthropic, Gemini, Mistral, and OpenAI
        database_uri=os.environ.get("MILVUS_URI"),  # Set the path (local) / url (remote) for the Milvus DB connection
        database_token=os.environ.get("MILVUS_TOKEN"),  # Optionally also set the access token
    )

    # Define inputs
    question: str = "What are the most recent research developments in the field of AI Safety?"
    breadth: int = 10
    depth: int = 2
    diversity: float = 0.0

    # Get result
    result: dict = generator.generate_answer_to_scientific_question(
        question=question, breadth=breadth, depth=depth, diversity=diversity
    )

    print(result["question_answer"])
    print(result["citations"])
