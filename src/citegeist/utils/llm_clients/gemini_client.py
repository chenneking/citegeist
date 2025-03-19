"""
Gemini API client implementation.
"""

import requests

from .base_client import LLMClient, exponential_backoff_retry


class GeminiClient(LLMClient):
    def __init__(
        self,
        api_key: str,
        model_name: str,
        embedding_model_name: str,  # Options: gemini-embedding-exp-03-07, text-embedding-004, embedding-001
    ):
        """
        Initializes the Gemini client.

        Args:
            model_name (str): The model name.
            api_key (str): The API key.
        """
        self.api_key = api_key
        self.model_name = model_name
        self.embedding_model_name = embedding_model_name

    """Client for Gemini API."""

    def get_completions(self, prompt: str, max_tokens: int = 4096, temperature: float = 0.0, **kwargs) -> str:
        """
        Wraps prompt in a contents object, which is passed to get_chat_completions.

        Args:
            prompt (str): The prompt.
            max_tokens (int): The maximum number of tokens to return.
            temperature (float): The temperature to use.

        Returns:
            The completions.
        """
        contents = [{"parts": [{"text": prompt}]}]
        return self.get_chat_completions(contents, max_tokens, temperature, **kwargs)

    def get_chat_completions(
        self, contents: list[dict[str, list[dict]]], max_tokens: int = 4096, temperature: float = 0.0, **kwargs
    ) -> str:
        """
        Gets chat completions from Gemini API based on the provided contents.

        Args:
            contents: The contents to get chat completions from.
            max_tokens (int): The maximum number of tokens to return.
            temperature (float): The temperature to use.

        Returns:
            The chat completions.
        """

        def call_model() -> str:
            headers = {"Content-Type": "application/json"}

            payload = {"contents": contents, "temperature": temperature, "max_output_tokens": max_tokens, **kwargs}

            request_url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}"
                f":generateContent?key={self.api_key}"
            )

            response = requests.post(url=request_url, headers=headers, json=payload)
            response.raise_for_status()
            reply = response.json()

            return reply["choices"][0]["message"]["content"]

        return exponential_backoff_retry(call_model)

    def get_embeddings(self, input_list: list[str], **kwargs) -> list[list[float]]:
        """
        Gets embeddings from Gemini for the provided texts.

        Args:
            input_list: List of text inputs to embed
            **kwargs: Additional model parameters

        Returns:
            List of embedding vectors
        """
        content = {"parts": [{"text": item} for item in input_list]}

        def call_model() -> str:
            headers = {"Content-Type": "application/json"}

            payload = {"model": self.embedding_model_name, "content": content, **kwargs}

            request_url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/{self.embedding_model_name}"
                f":embedContent?key={self.api_key}"
            )

            response = requests.post(url=request_url, headers=headers, json=payload)
            response.raise_for_status()
            reply = response.json()

            # TODO: figure out how this format changes when multiple text parts are provided as input
            return reply["embedding"]["values"]

        return exponential_backoff_retry(call_model)
        # TODO: Implement
