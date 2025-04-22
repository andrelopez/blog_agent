import pytest
from fastapi.testclient import TestClient
from blog_ingest.main import app
from unittest.mock import patch
import os

def load_html(filename):
    with open(filename, encoding='utf-8') as f:
        return f.read()

client = TestClient(app)

def test_ingest_endpoint():
    with patch("blog_ingest.main.fetch_and_parse_blogs", return_value=None):
        response = client.post("/ingest")
        assert response.status_code == 200
        data = response.json()
        assert "jobId" in data
        assert data["status"] == "started"

def test_ingest_endpoint_with_html():
    html_path = os.path.join(os.path.dirname(__file__), "test_html", "sample_bitovi_blog.html")
    html = load_html(html_path)
    with patch("blog_ingest.main.fetch_and_parse_blogs", return_value=None):
        response = client.post("/ingest")
        assert response.status_code == 200
        data = response.json()
        assert "jobId" in data
        assert data["status"] == "started" 