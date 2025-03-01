"""
Anthropic API client implementation.
"""

import requests
from typing import Dict, List
from .base_client import LLMClient, exponential_backoff_retry

class AnthropicClient(LLMClient):
    """Client for Anthropic API."""

    def __init__(
        self
    ):
        """
        Initialize the OpenAI client.

        Args:

        """
        # TODO: Implement
        pass

    def get_completions(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        **kwargs
    ) -> str:
        """

        """
        # TODO: Implement
        pass

    def get_chat_completions(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 4096,
        temperature: float = 0.0,
        **kwargs
    ) -> str:
        """

        """
        # TODO: Implement
        pass


    def get_embeddings(
            self,
            input_list: List[str],
            model: str = None,
            **kwargs
    ) -> List[List[float]]:
        """
        Gets embeddings from Anthropic for the provided texts.

        Args:
            input_list: List of text inputs to embed
            model: Override the default embedding model
            **kwargs: Additional model parameters

        Returns:
            List of embedding vectors
        """
        raise NotImplementedError("Anthropic doesn't support generating embeddings.")



