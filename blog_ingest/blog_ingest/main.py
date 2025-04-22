from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from .services.blog_service import fetch_and_parse_blogs
from .services.embedding_service import run_ingestion_and_embedding
from fastapi.staticfiles import StaticFiles
import uuid
import logging

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_parse_and_embed_blogs():
    fetch_and_parse_blogs()
    run_ingestion_and_embedding()

@app.post("/ingest")
def ingest(background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    logger.info(f"Ingestion job started. jobId={job_id}")
    background_tasks.add_task(fetch_parse_and_embed_blogs)
    return {"jobId": job_id, "status": "started"}

# Serve static files (including index.html)
app.mount("/", StaticFiles(directory="blog_ingest/static", html=True), name="static") 