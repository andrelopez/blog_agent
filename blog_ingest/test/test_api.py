import pytest
from fastapi.testclient import TestClient
from blog_ingest.main import app
from unittest.mock import patch

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