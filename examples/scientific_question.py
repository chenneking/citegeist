"""
Example demonstrating how to use the citegeist pipeline to generate an answer to a scientific question.
"""

from citegeist.generator import Generator

if __name__ == "__main__":
    # Setup Generator
    generator = Generator(
        llm_provider="gemini",  # choice of: Azure (OpenAI Studio), Anthropic, Gemini, Mistral, and OpenAI
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
