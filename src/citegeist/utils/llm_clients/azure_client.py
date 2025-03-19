"""
Azure OpenAI API client implementation.
"""

import requests

from .base_client import LLMClient, exponential_backoff_retry


class AzureClient(LLMClient):
    """Client for Azure OpenAI API."""

    def __init__(
        self,
        api_key: str,
        endpoint: str,
        deployment_id: str = None,
        api_version: str = None,
        embedding_deployment_id: str = None,
        embedding_api_version: str = None,
    ):
        """
        Initialize the Azure OpenAI client.

        Args:
            api_key: Azure API key
            endpoint: Azure API endpoint (without https:// or .openai.azure.com)
            deployment_id: Deployment ID for completions/chat (equivalent to model_name)
            api_version: API version to use
            embedding_deployment_id: Deployment ID for embeddings (equivalent to embedding_model_name)
            embedding_api_version: Embedding API version to use
        """
        self.endpoint = endpoint
        self.api_key = api_key
        self.deployment_id = deployment_id
        self.api_version = api_version
        self.embedding_deployment_id = embedding_deployment_id
        self.embedding_api_version = embedding_api_version

    def get_completion(self, prompt: str, max_tokens: int = 4096, temperature: float = 0.0, **kwargs) -> str:
        """
        Gets completions from Azure OpenAI based on a prompt.

        Args:
            prompt: The input prompt text
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 = deterministic)
            **kwargs: Additional model parameters

        Returns:
            The completion text
        """
        messages = [{"role": "user", "content": prompt}]
        return self.get_chat_completion(messages, max_tokens, temperature, **kwargs)

    def get_chat_completion(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 4096,
        temperature: float = 0.0,
        **kwargs,
    ) -> str:
        """
        Gets chat completions from Azure OpenAI based on a conversation.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 = deterministic)
            **kwargs: Additional model parameters

        Returns:
            The completion text
        """

        def call_model() -> str:
            headers = {
                "Content-Type": "application/json",
                "api-key": self.api_key,
            }

            payload = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens, **kwargs}

            request_url = (
                f"{self.endpoint}/openai/deployments/"
                f"{self.deployment_id}/chat/completions?api-version={self.api_version}"
            )

            response = requests.post(url=request_url, headers=headers, json=payload)
            response.raise_for_status()
            reply = response.json()

            return reply["choices"][0]["message"]["content"]

        return exponential_backoff_retry(call_model)

    def get_embeddings(self, input_list: list[str], **kwargs) -> list[list[float]]:
        """
        Gets embeddings from Azure OpenAI for the provided texts.

        Args:
            input_list: List of text inputs to embed
            **kwargs: Additional model parameters

        Returns:
            List of embedding vectors
        """

        def call_model() -> list[list[float]]:
            headers = {
                "Content-Type": "application/json",
                "api-key": self.api_key,
            }

            payload = {"input": input_list, **kwargs}

            request_url = (
                f"https://{self.endpoint}.openai.azure.com/openai/deployments/"
                f"{self.embedding_deployment_id}/embeddings?api-version={self.api_version}"
            )

            response = requests.post(url=request_url, headers=headers, json=payload)
            response.raise_for_status()
            json_response = response.json()

            # Extract the embedding vectors
            embeddings = [item["embedding"] for item in json_response["data"]]
            return embeddings

        return exponential_backoff_retry(call_model)
