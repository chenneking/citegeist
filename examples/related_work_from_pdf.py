"""
Example demonstrating how to use the citegeist pipeline to generate a related works section
 (based on a pdf paper as input).
"""

import os

from citegeist.generator import Generator
from citegeist.utils.citations import (
    extract_text_by_page_from_pdf,
    remove_citations_and_supplements,
)

if __name__ == "__main__":
    # Setup Generator
    generator = Generator(
        llm_provider="gemini",  # choice of: Azure (OpenAI Studio), Anthropic, Gemini, Mistral, and OpenAI
        database_uri=os.environ.get("MILVUS_URI"),  # Set the path (local) / url (remote) for the Milvus DB connection
        database_token=os.environ.get("MILVUS_TOKEN"),  # Optionally also set the access token
    )

    # Define inputs
    # TODO: PDF example
    path_to_pdf: str = ""
    pdf_content: bytes = None
    raw_pages = extract_text_by_page_from_pdf(pdf_content)
    pages: list[str] = remove_citations_and_supplements(raw_pages)
    breadth: int = 10
    depth: int = 2
    diversity: float = 0.0

    # Get result
    result: dict = generator.generate_related_work_from_paper(
        pages=pages, breadth=breadth, depth=depth, diversity=diversity
    )

    print(result["related_works"])
    print(result["citations"])
