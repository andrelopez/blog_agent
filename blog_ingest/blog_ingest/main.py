from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from .services.blog_service import fetch_and_parse_blogs
from .services.embedding_service import hybrid_search, generate_rag_answer
from fastapi.staticfiles import StaticFiles

app = FastAPI()

class AnswerRequest(BaseModel):
    question: str

@app.post("/ingest")
def ingest():
    try:
        result = fetch_and_parse_blogs()
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

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