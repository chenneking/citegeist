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
- Python 3.8+
- Dependencies listed in `requirements.txt`
- Access to arXiv's metadata and API

### Setup
```bash
git clone https://github.com/chenneking/citegeist.git
cd citegeist
pip install -r requirements.txt
```
Setup the milvus database. (TODO: explain this step)

## Usage

### Generating Related Work Section
To generate a related work section for a given abstract:
```python
TODO
```

### Running the Web Interface
Citegeist also provides a **web-based interface** to input abstracts or upload full papers.
```bash
cd webapp
fastapi run server.py
```
Then, access the UI at `http://localhost:8000`.

### Web-UI
![Web-UI Overview](https://github.com/chenneking/citegeist/blob/main/img/citegeist.jpg?raw=true)


## Customization
Citegeist allows users to adjust three key parameters:
- **Breadth**`n`: Number of candidate papers retrieved.
- **Depth**`k`: Number of relevant pages extracted from each paper.
- **Diversity**`w`: Balancing factor between similarity and variety in retrieved papers.
The parameters can either be set in the API calls in Python, or when using the Web-Interface.
