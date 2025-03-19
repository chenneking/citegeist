"""
Azure OpenAI API client implementation.
"""

import requests
from typing import Dict, List
from .base_client import LLMClient, exponential_backoff_retry

class AzureClient(LLMClient):
    """Client for Azure OpenAI API."""
    
    def __init__(
        self, 
        endpoint: str, 
        api_key: str, 
        deployment_id: str = None,
        embedding_deployment_id: str = None,
        api_version: str = "2023-05-15"
    ):
        """
        Initialize the Azure OpenAI client.
        
        Args:
            endpoint: Azure API endpoint (without https:// or .openai.azure.com)
            api_key: Azure API key
            deployment_id: Deployment ID for completions/chat
            embedding_deployment_id: Deployment ID for embeddings (falls back to deployment_id if not provided)
            api_version: API version to use
        """
        self.endpoint = endpoint
        self.api_key = api_key
        self.deployment_id = deployment_id
        self.embedding_deployment_id = embedding_deployment_id or deployment_id
        self.api_version = api_version
    
    def get_completions(
        self, 
        prompt: str, 
        max_tokens: int = 4096, 
        temperature: float = 0.0,
        **kwargs
    ) -> str:
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
        return self.get_chat_completions(messages, max_tokens, temperature, **kwargs)
    
    def get_chat_completions(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 4096,
        temperature: float = 0.0,
        deployment_id: str = None,
        **kwargs
    ) -> str:
        """
        Gets chat completions from Azure OpenAI based on a conversation.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 = deterministic)
            deployment_id: Override the default deployment ID
            **kwargs: Additional model parameters
            
        Returns:
            The completion text
        """
        deployment = deployment_id or self.deployment_id
        
        if not deployment:
            raise ValueError("No deployment ID provided for chat completions")
        
        def call_model() -> str:
            headers = {
                "Content-Type": "application/json",
                "api-key": self.api_key,
            }
            
            payload = {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs
            }
            
            request_url = (
                f"https://{self.endpoint}.openai.azure.com/openai/deployments/"
                f"{deployment}/chat/completions?api-version={self.api_version}"
            )
            
            response = requests.post(url=request_url, headers=headers, json=payload)
            response.raise_for_status()
            reply = response.json()
            
            return reply["choices"][0]["message"]["content"]
        
        return exponential_backoff_retry(call_model)
    
    def get_embeddings(
        self, 
        input_list: List[str],
        deployment_id: str = None,
        **kwargs
    ) -> List[List[float]]:
        """
        Gets embeddings from Azure OpenAI for the provided texts.
        
        Args:
            input_list: List of text inputs to embed
            deployment_id: Override the default embedding deployment ID
            **kwargs: Additional model parameters
            
        Returns:
            List of embedding vectors
        """
        deployment = deployment_id or self.embedding_deployment_id
        
        if not deployment:
            raise ValueError("No deployment ID provided for embeddings")
        
        def call_model() -> List[List[float]]:
            headers = {
                "Content-Type": "application/json",
                "api-key": self.api_key,
            }
            
            payload = {
                "input": input_list,
                **kwargs
            }
            
            request_url = (
                f"https://{self.endpoint}.openai.azure.com/openai/deployments/"
                f"{deployment}/embeddings?api-version={self.api_version}"
            )
            
            response = requests.post(url=request_url, headers=headers, json=payload)
            response.raise_for_status()
            json_response = response.json()
            
            # Extract the embedding vectors
            embeddings = [item["embedding"] for item in json_response["data"]]
            return embeddings
        
        return exponential_backoff_retry(call_model)