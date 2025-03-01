"""
LLM API Client Module

This module provides a collection of clients for interacting with different LLM APIs.
"""

from .base_client import LLMClient
from .azure_client import AzureClient
from .mistral_client import MistralClient
from .gemini_client import GeminiClient

# Factory function to create appropriate client
def create_client(provider: str, **kwargs) -> LLMClient:
    """
    Factory function to create an LLM client based on the provider.
    
    Args:
        provider: The LLM provider name ('azure', 'openai', 'anthropic')
        **kwargs: Provider-specific configuration arguments
        
    Returns:
        An instance of the appropriate LLM client
        
    Raises:
        ValueError: If the provider is not supported
    """
    provider = provider.lower()
    
    if provider == 'azure':
        return AzureClient(**kwargs)
    elif provider == 'gemini':
        return GeminiClient(**kwargs)
    elif provider == 'mistral':
        return MistralClient(**kwargs)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")