import asyncio
import logging
import os
import uuid
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from citegeist import Generator
from citegeist.utils.citations import (
    extract_text_by_page_from_pdf,
    remove_citations_and_supplements,
)

logger = logging.getLogger(__name__)


class Item(BaseModel):
    abstract: str
    breadth: int
    depth: int
    diversity: int


class JobStatus(BaseModel):
    status: str
    progress: int = 0
    status_text: str = ""
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# Load environment variables
load_dotenv()

# Setup citegeist Generator
generator = Generator(
    llm_provider="gemini",
    database_uri=os.environ.get("MILVUS_URI"),
    database_token=os.environ.get("MILVUS_TOKEN"),
    api_key=os.getenv("GEMINI_API_KEY"),
    model_name="gemini-2.0-flash",
)

# FastAPI logic
app = FastAPI(title="Citegeist", summary="Delivers a related works section and all the included citations.")

# Mount static files directory for assets (CSS, JS)
app.mount("/static", StaticFiles(directory="./static"), name="static")

# In-memory job storage
jobs: Dict[str, JobStatus] = {}


@app.middleware("http")
async def maintenance_mode_middleware(request: Request, call_next):
    # Check if maintenance mode is active
    if os.getenv("MAINTENANCE_MODE", False):
        # Allow static files to be served so that CSS/JS and other assets load
        if request.url.path.startswith("/static"):
            return await call_next(request)

        # For the front page, return the maintenance.html file
        if request.url.path == "/":
            return FileResponse("static/maintenance.html", status_code=503)
        else:
            # For all other routes, return an error response
            return JSONResponse(status_code=503, content={"message": "Maintenance Mode Active"})
    return await call_next(request)


@app.get("/")
def frontpage():
    return FileResponse("static/index.html")


@app.post("/create-job")
async def create_job(
    breadth: int = Form(...),
    depth: int = Form(...),
    diversity: float = Form(...),
    abstract: Optional[str] = Form(None),
    pdf: Optional[UploadFile] = File(None),
    background_tasks: BackgroundTasks = None,
):
    # Validate input
    if not abstract and not pdf:
        raise HTTPException(status_code=400, detail="Either abstract text or PDF file must be provided")

    if not (5 <= breadth <= 20) or not (1 <= depth <= 5) or not (0 <= diversity <= 1):
        raise HTTPException(status_code=400, detail="Invalid breadth, depth, or diversity value")

    # Create a new job
    job_id: uuid = str(uuid.uuid4())
    jobs[job_id] = JobStatus(status="created")

    # Read PDF content if provided
    pdf_pages: list[str] = None
    if pdf:
        try:
            pdf_content = await pdf.read()
            raw_pages = extract_text_by_page_from_pdf(pdf_content)
            pdf_pages = remove_citations_and_supplements(raw_pages)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading PDF: {str(e)}")

    # Start processing in background
    background_tasks.add_task(process_job, job_id, breadth, depth, diversity, abstract, pdf_pages)

    # Return job ID for status polling
    return {"job_id": job_id}


@app.get("/status/{job_id}")
def status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    # Return the status, progress percentage, and status text
    return {
        "status": job.status,
        "progress": job.progress,
        "status_text": job.status_text,
        "result": job.result,
        "error": job.error,
    }


async def process_job(
    job_id: str,
    breadth: int,
    depth: int,
    diversity: float,
    abstract: Optional[str] = None,
    pdf_pages: Optional[list[str]] = None,
):
    def status_callback(step, status_text):
        progress: int = int((step / 8) * 100)
        jobs[job_id].progress = progress
        jobs[job_id].status_text = status_text
        logger.info(f"Job {job_id}: Step {step}, Status: {status_text}")

    try:
        # Initialize job status
        jobs[job_id].status = "processing"
        jobs[job_id].progress = 0
        jobs[job_id].status_text = "Starting job."

        # Trigger correct logic based on provided input
        result: Optional[Dict[str, Any]] = None
        if abstract:
            result: dict = await asyncio.to_thread(
                generator.generate_related_work,
                abstract=abstract,
                breadth=breadth,
                depth=depth,
                diversity=diversity,
                status_callback=status_callback,
            )

        elif pdf_pages:
            result: dict = await asyncio.to_thread(
                generator.generate_related_work_from_paper,
                pages=pdf_pages,
                breadth=breadth,
                depth=depth,
                diversity=diversity,
                status_callback=status_callback,
            )

        # Dummy job processing
        # result = await generator.dummy(status_callback)

        # Label job as completed (this is important for the frontend ajax logic)
        jobs[job_id].status = "completed"
        jobs[job_id].result = result

    except Exception as e:
        logger.error(f"Job {job_id} failed: {str(e)}", exc_info=True)
        # Handle exceptions
        jobs[job_id].status = "failed"
        jobs[job_id].error = str(e)
