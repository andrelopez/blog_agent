from fastapi import FastAPI
from fastapi.responses import JSONResponse
from blog_service import fetch_and_parse_blogs

app = FastAPI()

@app.post("/ingest")
def ingest():
    try:
        result = fetch_and_parse_blogs()
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500) 