"""
Example demonstrating how to use different LLM providers with Citegeist.
"""

import os

from dotenv import load_dotenv

from citegeist.utils.llm_clients import (
    AzureClient,
    GeminiClient,
    LLMClient,
    OpenAIClient,
)
from citegeist.utils.llm_clients.anthropic_client import AnthropicClient

# Load environment variables from .env file
load_dotenv()


def gemini() -> None:
    """
    Example use of the GeminiClient.
    """
    client: LLMClient = GeminiClient(
        api_key=os.getenv("GEMINI_API_KEY"),
        model_name="gemini-2.0-flash",
        embedding_model_name="gemini-embedding-exp-03-07",
    )
    print("Using Gemini:")
    print(client.get_completion("Hello World!"))
    print(
        client.get_chat_completion(
            [
                {"role": "user", "parts": [{"text": "Hello World!"}]},
                {"role": "model", "parts": [{"text": "Hi! Great to meet you!"}]},
                {"role": "user", "parts": [{"text": "What does the dog say?"}]},
            ]
        )
    )


def azure() -> None:
    """
    Example use of the AzureClient (via OpenAI Studio).
    """
    client: LLMClient = AzureClient(
        api_key=os.getenv("AZURE_API_KEY"),
        endpoint="https://cai-project.openai.azure.com",
        deployment_id="gpt-4o",  # this is equivalent to model_name for other clients (but Azure calls it that)
        api_version="2024-10-21",
        embedding_deployment_id="text-embedding-ada-002",  # this is equivalent to embedding_model_name
        embedding_api_version="2023-05-15",
    )
    print("Using Azure:")
    print(client.get_completion("Hello World!"))
    print(
        client.get_chat_completion(
            [
                {"role": "user", "content": "Hello World!"},
                {"role": "assistant", "content": "Hi! Great to meet you!"},
                {"role": "user", "content": "What does the dog say?"},
            ]
        )
    )


def anthropic() -> None:
    """
    Example use of the AnthropicClient.
    """
    client: LLMClient = AnthropicClient(
        api_key=os.getenv("ANTHROPIC_API_KEY"), model_name="claude-3-5-haiku-20241022", api_version="2023-06-01"
    )
    print("Using Anthropic:")
    print(client.get_completion("Hello World!"))
    print(
        client.get_chat_completion(
            [
                {"role": "user", "content": "Hello World!"},
                {"role": "assistant", "content": "Hi! Great to meet you!"},
                {"role": "user", "content": "What does the dog say?"},
            ]
        )
    )


def openai() -> None:
    """
    Example use of the OpenAIClient.
    """
    client: LLMClient = OpenAIClient(
        api_key=os.getenv("OPENAI_API_KEY"),
        endpoint="https://cai-project.openai.azure.com",
        deployment_id="gpt-4o",  # this is equivalent to model_name for other clients (but Azure calls it that)
        api_version="2024-10-21",
        embedding_deployment_id="text-embedding-ada-002",  # this is equivalent to embedding_model_name
        embedding_api_version="2023-05-15",
    )
    print("Using Azure:")
    print(client.get_completion("Hello World!"))
    print(
        client.get_chat_completion(
            [
                {"role": "user", "content": "Hello World!"},
                {"role": "assistant", "content": "Hi! Great to meet you!"},
                {"role": "user", "content": "What does the dog say?"},
            ]
        )
    )


if __name__ == "__main__":
    anthropic()
