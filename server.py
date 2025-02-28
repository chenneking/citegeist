import io
import os
import uuid
import asyncio
from typing import Optional, Dict, Any
from fastapi import FastAPI, Response, Form, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from generator import generate_related_work, generate_related_work_from_paper
from utils.citations import extract_text_by_page_from_pdf, remove_citations_and_supplements


class Item(BaseModel):
    abstract: str
    breadth: int
    depth: int
    diversity: int


app = FastAPI()

# Mount static files directory for assets (CSS, JS)
app.mount("/static", StaticFiles(directory="./static"), name="static")

# In-memory job storage
jobs: Dict[str, JobStatus] = {}


@app.get("/")
def frontpage():
    return FileResponse("./static/index.html")


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