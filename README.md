> **⚡ TL;DR:** Citegeist is currently live at [https://citegeist.org](https://citegeist.org). Feel free to use it, but if you want to use the Python API, read the instructions below.
# Citegeist: Automated Generation of Related Work Analysis on the arXiv Corpus

Citegeist is an automated system that generates related work sections and citation-backed outputs using **Retrieval-Augmented Generation (RAG)** on the **arXiv corpus**. It leverages **embedding-based similarity search**, **multi-stage filtering**, and **summarization** to retrieve and synthesize the most relevant sources. Citegeist is designed to help researchers integrate factual and up-to-date references into their work.

A preprint describing the system in detail can be found here: [arXiv (TODO)]()


## Features
- **Automated related work generation** based on abstract similarity matching.
- **Multi-stage retrieval pipeline** using embedding-based similarity search.
- **Summarization and synthesis** of retrieved papers to generate a well-structured related works section.
- **Customizable parameters** for breadth, depth, and diversity of retrieved papers.
- **Efficient database updates** to incorporate newly published arXiv papers.

## Installation

### Requirements
- Python 3.12
- Access to arXiv's metadata and API

### Setup (Regular Users)
1. Install the citgeist package.
    ```bash
    pip install citegeist
    ```
2. Setup the Milvus database. As of March 2025, we provide a hosted version of this database that you can use for free (see usage instructions below). If we discontinue this, please refer to the additional information provided in the respective sections below.
3. Run the pipeline.

### Setup (Web-Interface)
You only need to follow these steps if you want to use the web-interface! Using the setup steps above are sufficient if you wish to use the python interface.
1. Clone the repo
   ```bash
   git clone https://github.com/chenneking/citegeist.git
   cd citegeist
   ```
2. Optional (but recommended): Create and activate a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install development dependencies
   ```bash
   pip install -e .[webapp] # if you're using uv: pip install -e ."[webapp]" 
   ```

### Setup (Developers)
If you wish to work on/modify the core citegeist code, please use the following setup steps.
1. Clone the repo
   ```bash
   git clone https://github.com/chenneking/citegeist.git
   cd citegeist
   ```
2. Optional (but recommended): Create and activate a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install development dependencies
   ```bash
   pip install -e .[dev] # if you're using uv: pip install -e ."[dev]" 
   ```

## Usage

### Customization
Citegeist allows users to adjust three key parameters:
- **Breadth**`n`: Number of candidate papers retrieved.
- **Depth**`k`: Number of relevant pages extracted from each paper.
- **Diversity**`w`: Balancing factor between similarity and variety in retrieved papers.
The parameters can either be set in the API calls in Python, or when using the Web-Interface.


### Generating Related Work Section
As of March 2025, we provide a hosted Milvus database that you can use by setting the following environment variables:
```dotenv
MILVUS_URI=http://49.12.219.90:19530
MILVUS_TOKEN=citegeist:citegeist
```
To generate a related work section for a given abstract:
```python
from citegeist import Generator
import os

generator = Generator(
   llm_provider="gemini",  # Choice of: "azure" (OpenAI Studio), "anthropic", "gemini", "mistral", and "openai"
   api_key=os.environ.get("GEMINI_API_KEY"), # Here, you will need to set the respective API key
   model_name="gemini-2.0-flash", # Choose the model that the provider supports
   database_uri=os.environ.get("MILVUS_URI"),  # Set the path (local) / url (remote) for the Milvus DB connection
   database_token=os.environ.get("MILVUS_TOKEN"),  # Optionally, also set the access token (you DON'T need to set this when using the locally hosted Milvus Database)
)
# Define input abstract and breadth (5-20), depth (1-5), and diversity (0.0-1.0) parameters.
abstract = "..."
breadth = 10
depth = 2
diversity = 0.0
related_works, citations = generator.generate_related_work(abstract, breadth, depth, diversity)
```
**Local Deployment:**

If you, however, wish to run the Milvus database locally, you can do so by downloading [database.db](https://huggingface.co/datasets/chenneking/citegeist-milvus-db/blob/main/database.db), and provide an (absolute) path to the file as the value for `MILVUS_URI`. You **don't** have to set `MILVUS_TOKEN` for this.
```dotenv
MILVUS_URI=<path_to_database.db_goes_here>
```
Then, to generate a related work section for a given abstract:
```python
from citegeist import Generator
import os

generator = Generator(
   llm_provider="gemini",  # Choice of: "azure" (OpenAI Studio), "anthropic", "gemini", "mistral", and "openai"
   api_key=os.environ.get("GEMINI_API_KEY"), # Here, you will need to set the respective API key
   model_name="gemini-2.0-flash", # Choose the model that the provider supports
   database_uri=os.environ.get("MILVUS_URI"),  # Set the path (local) database.db file
)
# Define input abstract and breadth (5-20), depth (1-5), and diversity (0.0-1.0) parameters.
abstract = "..."
breadth = 10
depth = 2
diversity = 0.0
related_works, citations = generator.generate_related_work(abstract, breadth, depth, diversity)
```
Please refer to [examples](/examples) for more usage examples.

### Running the Web Interface
Beyond the python interface, citegeist also provides a **web-based interface** to input abstracts or upload full papers. 

**Option 1: citegeist.org**

As of March 2025, we provide a hosted version available [here](https://citegeist.org/).

However, if you prefer to run this locally and have installed the additional `[webapp]` requirements as described above, you have the following options:

**Option 2: uvicorn**

To start the web-interface using uvicorn:
```bash
uvicorn webapp.server:app --reload
```
Then, access the UI at `http://localhost:8000`.

**Option 3: Docker**

If you prefer to use docker, build the image:
```bash
docker build -t citegeist-webapp -f src/webapp/Dockerfile .
```
Run a container with the image:
```bash
docker run -p 80:80 citegeist-webapp
```
Then, access the UI at `http://localhost`.


### Web-UI
![Web-UI Overview](https://github.com/chenneking/citegeist/blob/main/img/citegeist.jpg?raw=true)
