# Stage 1: Build and test
FROM python:3.11-slim AS test
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY blog_ingest ./blog_ingest
ENV PYTHONPATH=/app
ARG TEST_TIMESTAMP=manual
RUN echo "=== RUNNING TESTS ===" && pytest blog_ingest/test_pkg -v && echo "=== TESTS COMPLETE ==="

# Stage 2: Production image (only if tests pass)
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY blog_ingest ./blog_ingest
ENV PYTHONPATH=/app
# Default: run FastAPI app
CMD ["uvicorn", "blog_ingest.main:app", "--host", "0.0.0.0", "--port", "8000"] 