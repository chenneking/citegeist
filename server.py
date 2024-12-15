import io
from typing import Optional
from fastapi import FastAPI, Response, Form, UploadFile, File
from pydantic import BaseModel
from generator import generate_related_work, generate_related_work_from_paper
from utils.citations import extract_text_by_page_from_pdf, remove_citations_and_supplements


class Item(BaseModel):
    abstract: str
    breadth: int
    depth: int
    diversity: int


app = FastAPI()


@app.get("/")
def frontpage():
    with open("./frontend/index.html") as f:
        data = f.read()
    return Response(content=data, media_type="text/html")


@app.post("/generate")
async def generate(
        breadth: int = Form(...),
        depth: int = Form(...),
        diversity: float = Form(...),
        abstract: Optional[str] = Form(None),
        pdf: Optional[UploadFile] = File(None)
):
    # Validate input
    if not abstract and not pdf:
        raise ValueError("Either abstract text or PDF file must be provided")

    # Extract text from PDF if uploaded
    if pdf:
        try:
            content = await pdf.read()
            raw_pages = extract_text_by_page_from_pdf(content)
            pages = remove_citations_and_supplements(raw_pages)

            return generate_related_work_from_paper(
                pages,
                breadth,
                depth,
                diversity
            )
        except Exception as e:
            raise ValueError(f"Error processing PDF: {str(e)}")
    # Generate related work if abstract is provided
    else:
        return generate_related_work(
            abstract,
            breadth,
            depth,
            diversity
        )