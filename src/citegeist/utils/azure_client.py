import logging
import time
from typing import Callable
import requests
import random


class AzureClient:
    def __init__(self, endpoint: str, deployment_id: str, api_key: str):
        """
        Initialize the AzureClient object.
        :param endpoint: azure api endpoint
        :param deployment_id: azure api deployment id
        :param api_key: azure api key
        """
        self.endpoint = endpoint
        self.deployment_id = deployment_id
        self.api_key = api_key

    def get_completions(
        self, prompt: str, api_version: str, max_tokens: int = 4096
    ) -> str:
        """
        Prompts an OpenAI LLM and returns it's output.
        :param prompt: user prompt
        :param api_version: azure api model version
        :param max_tokens: max number of output tokens
        :return: LLM output string
        """

        def call_model() -> str:
            """
            Defines the HTTP request to trigger a model response in Azure
            :return: Model output
            """
            headers: dict = {
                "Content-Type": "application/json",
                "api-key": f"{self.api_key}",
            }

            payload: dict = {
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,  # Deterministic output
                "top_p": 1.0,  # Full sampling
                "max_tokens": max_tokens,
            }

            request_url: str = (
                f"https://{self.endpoint}.openai.azure.com/openai/deployments/{self.deployment_id}/chat/completions?api-version={api_version}"
            )

            response = requests.post(url=request_url, headers=headers, json=payload)

            response.raise_for_status()  # Raise an HTTPError for bad responses
            reply = response.json()

            # Extracting the assistant's response
            return reply["choices"][0]["message"]["content"]

        return exponential_backoff_retry(call_model)

    def get_embeddings(self, input_list: list[str], api_version: str) -> list[float]:
        """
        Prompts an OpenAI embedding model and returns it's output.
        :param input_list: list of tokens
        :param api_version: azure api model version
        :return:
        """

        def call_model() -> list[float]:
            headers: dict = {
                "Content-Type": "application/json",
                "api-key": f"{self.api_key}",
            }

            payload: dict = {
                "input": input_list,
            }

            request_url: str = (
                f"https://{self.endpoint}.openai.azure.com/openai/deployments/{self.deployment_id}/embeddings?api-version={api_version}"
            )

            response = requests.post(url=request_url, headers=headers, json=payload)

            response.raise_for_status()
            json_response = response.json()
            return json_response["data"]

        return exponential_backoff_retry(call_model)


def exponential_backoff_retry(
    func: Callable, retries: int = 10, backoff_factor: int = 2, max_wait: int = 120
):
    """
    Repeatedly calls a function with exponential backoff if it fails.
    :param func: callable function
    :param retries: max number of retries before giving up
    :param backoff_factor: factor by which exponential retry is increased
    :param max_wait: max number of seconds to sleep before retrying
    :return:
    """
    wait: int = 1
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:
            if "429" in str(e) or "529" in str(e):
                logging.warning(
                    f"Rate limit exceeded. Attempt {attempt + 1} of {retries}. Retrying in {wait} seconds."
                )
                time.sleep(wait + random.uniform(0, 1))  # Adding jitter
                wait = min(wait * backoff_factor, max_wait)
            else:
                raise e
    logging.error("Exceeded maximum retries due to rate limit.")
    raise Exception("Exceeded maximum retries due to rate limit.")