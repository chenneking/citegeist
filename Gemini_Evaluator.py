import google
from google.cloud import aiplatform_v1beta1 as aiplatform
from google.oauth2 import service_account
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Image, GenerationConfig
import logging
import time
import random

# Path to your service account key file
key_path = "/Users/carl/Downloads/stellar_depth_account_1.json"

# Create credentials using the service account key
credentials = service_account.Credentials.from_service_account_file(key_path)

# Initialize Google Cloud AI Platform
google.cloud.aiplatform.init(project="stellar-depth-441503-q5", location="us-central1", credentials=credentials)

def prompt_gemini(project_id: str, model_name: str, prompt: str, temperature: float = 0., max_tokens: int = 4096):
    """Prompt the Gemini model via Google Vertex AI."""
    vertexai.init(project=project_id, location="us-central1")
    model = GenerativeModel(model_name)
        
    response = model.generate_content(prompt, generation_config=GenerationConfig(temperature=temperature))
    
    return response.text

def prompt_gemini_with_backoff(project_id: str, model_name: str, prompt: str):
    """Wrap Gemini prompt with exponential backoff in case of rate limit."""
    def call_gemini():
        return prompt_gemini(project_id, model_name, prompt)
    return exponential_backoff_retry(call_gemini)

def exponential_backoff_retry(func, retries=10, backoff_factor=2, max_wait=120):
    """Retry with exponential backoff in case of rate limit error (429)."""
    wait = 1
    for attempt in range(retries):
        try:
            return func()  # Call the function
        except Exception as e:
            if "429" in str(e) or "529" in str(e):  # Rate limit errors
                logging.warning(f"Rate limit exceeded. Attempt {attempt + 1} of {retries}. Retrying in {wait} seconds.")
                time.sleep(wait + random.uniform(0, 1))  # Adding jitter to prevent thundering herd problem
                wait = min(wait * backoff_factor, max_wait)
            else:
                raise e
    logging.error("Exceeded maximum retries due to rate limit.")
    raise Exception("Exceeded maximum retries due to rate limit.")
    
