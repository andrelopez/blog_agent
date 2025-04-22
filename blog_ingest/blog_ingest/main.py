from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from .services.blog_service import fetch_and_parse_blogs
from .services.embedding_service import hybrid_search, generate_rag_answer
from fastapi.staticfiles import StaticFiles
import uuid
import logging

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnswerRequest(BaseModel):
    question: str

@app.post("/ingest")
def ingest(background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    logger.info(f"Ingestion job started. jobId={job_id}")
    background_tasks.add_task(fetch_and_parse_blogs)
    return {"jobId": job_id, "status": "started"}

@app.post("/answer")
def answer(request: AnswerRequest):
    question = request.question
    try:
        articles = hybrid_search(question, top_k=20)
        answer_text = generate_rag_answer(question, articles)
        return JSONResponse({
            "question": question,
            "answer": answer_text,
            "references": [
                {
                    "title": a.get("title"),
                    "url": a.get("url"),
                    "score": a.get("score"),
                    "snippet": a.get("text")[:300],
                    "date_published": a.get("date_published"),
                    "author": a.get("author"),
                    "description": a.get("description"),
                }
                for a in articles
            ]
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# Serve static files (including index.html)
app.mount("/", StaticFiles(directory="blog_ingest/static", html=True), name="static") 