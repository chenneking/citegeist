import arxiv
import requests
import os
import time
import fitz

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

def download_pdf(arxiv_id, save_path="paper.pdf", retries=3):
    """
    Download the PDF from arXiv using the arXiv ID with retry logic.
    """
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    
    for attempt in range(retries):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an error for bad status codes
            
            # Download in chunks and write to file
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):  # 1 KB chunks
                    if chunk:
                        f.write(chunk)
            
            print(f"PDF downloaded successfully: {save_path}")
            return True  # Success
        except (requests.exceptions.RequestException, requests.exceptions.ChunkedEncodingError) as e:
            print(f"Download failed, attempt {attempt + 1}/{retries}: {e}")
            time.sleep(2)  # Wait a little before retrying
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return False
    
    print("Failed to download PDF after several attempts.")
    return False

def extract_text_by_page(pdf_path):
    """
    Extract text from each page of the downloaded PDF using PyMuPDF (fitz).
    """
    doc = fitz.open(pdf_path)
    pages_text = []
    
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)  # Load each page
        pages_text.append(page.get_text())  # Extract text from the page
    
    return pages_text

def process_arxiv_paper(arxiv_id):
    """
    Full process to download the paper, extract text, and delete the PDF.
    """
    pdf_path = "paper.pdf"  # Temporary file to store the downloaded PDF
    
    # Step 1: Download the PDF
    if not download_pdf(arxiv_id, pdf_path):
        return []  # Return empty if download fails
    
    # Step 2: Extract text from the PDF
    pages_text = extract_text_by_page(pdf_path)
    
    # Step 3: Delete the PDF file after extracting text
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
        print(f"PDF file deleted: {pdf_path}")
    
    return pages_text

