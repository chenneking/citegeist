"""
Mistral API client implementation.
"""

from typing import Dict, List

from .base_client import LLMClient


class MistralClient(LLMClient):
    """Client for Mistral API."""

    def __init__(self):
        """
        Initialize the Mistral client.

        Args:

        """
        # TODO: Implement
        pass

    def get_completion(self, prompt: str, max_tokens: int = 4096, temperature: float = 0.0, **kwargs) -> str:
        """ """
        # TODO: Implement
        pass

    def get_chat_completion(
        self, messages: List[Dict[str, str]], max_tokens: int = 4096, temperature: float = 0.0, **kwargs
    ) -> str:
        """ """
        # TODO: Implement
        pass
