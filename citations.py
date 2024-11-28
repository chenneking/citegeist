import arxiv

def get_arxiv_citation(arxiv_id):
    # Use the Client for fetching paper details
    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])
    paper = next(client.results(search), None)
    
    if not paper:
        return f"No paper found for arXiv ID: {arxiv_id}"
    
    # Format the citation (e.g., APA style)
    authors = ', '.join(author.name for author in paper.authors)
    title = paper.title
    year = paper.published.year
    return f"{authors} ({year}). {title}. arXiv:{arxiv_id}. https://arxiv.org/abs/{arxiv_id}"

def get_arxiv_abstract(arxiv_id):
    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])
    paper = next(client.results(search), None)
    
    if not paper:
        return f"No paper found for arXiv ID: {arxiv_id}"
    return paper.summary

def get_arxiv_publication_date(arxiv_id):
    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])
    paper = next(client.results(search), None)
    
    if not paper:
        return f"No paper found for arXiv ID: {arxiv_id}"
    return paper.published
    

# Example usage
arxiv_id = "2408.13001"
citation = get_arxiv_citation(arxiv_id)
print(citation)
