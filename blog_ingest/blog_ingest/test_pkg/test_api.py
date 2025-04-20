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
    mock_result = {
        "count": 1,
        "sample_articles": [
            {
                "url": "https://www.bitovi.com/blog/test-article",
                "title": "Test",
                "date_published": "2024-03-07T14:00:00",
                "date_modified": None,
                "author": "Author",
                "description": "desc",
                "text": "Test content",
                "text_length": 12,
                "text_preview": "Test content"
            }
        ]
    }
    with patch("blog_ingest.main.fetch_and_parse_blogs", return_value=mock_result):
        response = client.post("/ingest")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert data["sample_articles"][0]["title"] == "Test"

def test_ingest_endpoint_with_html():
    html_path = os.path.join(os.path.dirname(__file__), "test_html", "sample_bitovi_blog.html")
    html = load_html(html_path)
    # Simulate the parsed result as would be returned by fetch_and_parse_blogs
    mock_result = {
        "count": 1,
        "sample_articles": [
            {
                "url": "https://www.bitovi.com/blog/test-article",
                "title": "Comparing Schema Validation Libraries: AJV, Joi, Yup, and Zod",
                "date_published": "2023-02-03T19:26:35",
                "date_modified": None,
                "author": "Roy Ayoola",
                "description": "Find the best schema validator for your needs. Our comprehensive comparison of AJV, Joi, Yup, and Zod as schema validators for your Node.js project.",
                "tags": ["Backend", "node.js"],
                "text": "...",
                "text_length": 1000,
                "text_preview": "..."
            }
        ]
    }
    with patch("blog_ingest.main.fetch_and_parse_blogs", return_value=mock_result):
        response = client.post("/ingest")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        article = data["sample_articles"][0]
        assert article["title"] == "Comparing Schema Validation Libraries: AJV, Joi, Yup, and Zod"
        assert article["author"] == "Roy Ayoola"
        assert "schema validator" in article["description"]
        assert any("backend" in t.lower() for t in article["tags"])
        assert "2023" in article["date_published"] 