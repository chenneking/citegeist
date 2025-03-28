# Citegeist: Automated Generation of Related Work Analysis on the arXiv Corpus

Citegeist is an automated system that generates related work sections and citation-backed outputs using **Retrieval-Augmented Generation (RAG)** on the **arXiv corpus**. It leverages **embedding-based similarity search**, **multi-stage filtering**, and **summarization** to retrieve and synthesize the most relevant sources. Citegeist is designed to help researchers integrate factual and up-to-date references into their work.

A preprint describing the system in detail can be found here: [arXiv link (todo)]()


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
1. Install the citgeist package
    ```bash
    pip install citegeist
    ```
2. Setup the Milvus database. As of March 2025, we provide a hosted version of this database that you can use for free (see usage instructions below). If we discontinue this, or you prefer to run this locally, you can download the database as file here: [Huggingface](https://huggingface.co/datasets/chenneking/citegeist-milvus-db) and refer to it when using the generator.
3. Run the pipeline

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
To generate a related work section for a given abstract:

```python
from citegeist import Generator
import os

generator = Generator(
   llm_provider="gemini",  # choice of: "azure" (OpenAI Studio), "anthropic", "gemini", "mistral", and "openai"
   api_key=os.environ.get("GEMINI_API_KEY"),
   model_name="gemini-2.0-flash",
   database_uri=os.environ.get("MILVUS_URI"),  # Set the path (local) / url (remote) for the Milvus DB connection
   database_token=os.environ.get("MILVUS_TOKEN"),  # Optionally also set the access token (you DON'T need to set this when using the locally hosted Milvus Database)
)
# Define input abstract and breadth (5-20), depth (1-5), and diversity (0.0-1.0) parameters.
abstract = "..."
breadth = 10
depth = 2
diversity = 0.0
generator.generate_related_work(abstract, breadth, depth, diversity)
```
As of March 2025, we provide a hosted Milvus database that you can use by setting the following environment variables:
```dotenv
MILVUS_URI="http://49.12.219.90:19530"
MILVUS_TOKEN="citegeist:citegeist"
```
Please refer to examples/ for more usage examples.

### Running the Web Interface
Beyond the python interface, citegeist also provides a **web-based interface** to input abstracts or upload full papers. To start the web-interface:
```bash
uvicorn webapp.server:app --reload
```
Then, access the UI at `http://localhost:8000`.

### Web-UI
![Web-UI Overview](https://github.com/chenneking/citegeist/blob/main/img/citegeist.jpg?raw=true)
