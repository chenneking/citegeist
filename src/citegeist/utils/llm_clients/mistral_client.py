""" Mistral API client implementation. """
import os
from mistralai import Mistral
from .base_client import LLMClient, exponential_backoff_retry

class MistralClient(LLMClient):
    """Client for Mistral API."""
    
    def __init__(self, api_key: str, model_name: str = "mistral-large-latest"):
        """
        Initializes the Mistral client.
        
        Args:
            api_key (str): The API key.
            model_name (str): The model name. Default is "mistral-large-latest".
        """
        self.api_key = api_key
        self.model_name = model_name
        self.client = Mistral(api_key=api_key)
    
    def get_completion(self, prompt: str, max_tokens: int = 4096, temperature: float = 0.0, **kwargs) -> str:
        """
        Wraps prompt in a message object, which is passed to get_chat_completion.
        
        Args:
            prompt (str): The prompt.
            max_tokens (int): The maximum number of tokens to return.
            temperature (float): The temperature to use.
        
        Returns:
            The completion.
        """
        messages = [{"role": "user", "content": prompt}]
        return self.get_chat_completion(messages, max_tokens, temperature, **kwargs)
    
    def get_chat_completion(
        self, messages: list[dict[str, str]], max_tokens: int = 4096, temperature: float = 0.0, **kwargs
    ) -> str:
        """
        Gets chat completions from Mistral API based on the provided messages.
        
        Args:
            messages: The messages to get chat completions from.
            max_tokens (int): The maximum number of tokens to return.
            temperature (float): The temperature to use.
        
        Returns:
            The chat completion.
        """
        
        def call_model() -> str:
            chat_response = self.client.chat.complete(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            return chat_response.choices[0].message.content
        
        return exponential_backoff_retry(call_model)