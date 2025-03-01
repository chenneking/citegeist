import uuid
import asyncio
from typing import Optional, Dict, Any
from fastapi import FastAPI, Form, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel


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


app = FastAPI(
    title="Citegeist",
    summary="Delivers a related works section and all the included citations."
)

# Mount static files directory for assets (CSS, JS)
app.mount("/static", StaticFiles(directory="./static"), name="static")

# In-memory job storage
jobs: Dict[str, JobStatus] = {}


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
        background_tasks: BackgroundTasks = None
):
    # Validate input
    if not abstract and not pdf:
        raise HTTPException(status_code=400, detail="Either abstract text or PDF file must be provided")

    if not (5 <= breadth <= 20) or not (1 <= depth <= 5) or not (0 <= diversity <= 1):
        raise HTTPException(status_code=400, detail="Invalid breadth, depth, or diversity value")

    # Create a new job ID
    job_id: uuid = str(uuid.uuid4())
    
    # Initialize job status
    jobs[job_id] = JobStatus(
        status="created",
        progress=0,
        status_text="Starting..."
    )
    
    # Read PDF content if provided
    pdf_content: bytes = None
    if pdf:
        try:
            pdf_content = await pdf.read()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading PDF: {str(e)}")
    
    # Start processing in background
    background_tasks.add_task(
        process_job,
        job_id,
        breadth,
        depth,
        diversity,
        abstract,
        pdf_content
    )
    
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
        "error": job.error
    }


async def process_job(
        job_id: str,
        breadth: int,
        depth: int,
        diversity: float,
        abstract: Optional[str] = None,
        pdf_content: Optional[bytes] = None
):
    try:
        # Initialize job status
        jobs[job_id].status = "processing"
        jobs[job_id].progress = 1
        jobs[job_id].status_text = "Initializing"

        await asyncio.sleep(5)
        jobs[job_id].status = "processing"
        jobs[job_id].progress = 10
        jobs[job_id].status_text = "Step 1"

        await asyncio.sleep(5)
        jobs[job_id].status = "processing"
        jobs[job_id].progress = 50
        jobs[job_id].status_text = "Step 2"

        await asyncio.sleep(5)
        jobs[job_id].status = "processing"
        jobs[job_id].progress = 80
        jobs[job_id].status_text = "Step 3"

        await asyncio.sleep(5)
        jobs[job_id].status = "completed"
        jobs[job_id].progress = 100
        jobs[job_id].status_text = "Generated related works section!"
        jobs[job_id].result = {
            'related_works': 'Dummy value',
            'citations': ['Citation 1', 'Citation 2']
        }
    except Exception as e:
        # Handle exceptions
        jobs[job_id].status = "failed"
        jobs[job_id].error = str(e)

