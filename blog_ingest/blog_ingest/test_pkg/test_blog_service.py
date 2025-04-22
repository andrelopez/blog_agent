import pytest
from blog_ingest.services.blog_service import normalize_date, fetch_and_parse_blogs
from unittest.mock import patch, MagicMock
import datetime
import os

def load_html(filename):
    with open(filename, encoding='utf-8') as f:
        return f.read()

# Test normalize_date
@pytest.mark.parametrize("input_date,expected", [
    ("2024-03-07 14:00:00", "2024-03-07T14:00:00"),
    ("March 7, 2024, 2:00:00 PM", "2024-03-07T14:00:00"),
    (None, None),
    ("not-a-date", "not-a-date"),
])
def test_normalize_date(input_date, expected):
    result = normalize_date(input_date)
    if expected is None or expected == "not-a-date":
        assert result == expected
    else:
        # Only compare up to seconds for ISO format
        assert result[:19] == expected

# Test fetch_and_parse_blogs with HTML mock file
@patch("blog_ingest.services.blog_service.requests.get")
@patch("blog_ingest.services.blog_service.get_pg_connection")
@patch("blog_ingest.services.blog_service.ensure_blog_articles_table")
@patch("blog_ingest.services.blog_service.insert_blog_article")
def test_fetch_and_parse_blogs_with_html(mock_insert, mock_ensure, mock_conn, mock_requests):
    sitemap_xml = '''<?xml version="1.0" encoding="UTF-8"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://www.bitovi.com/blog/test-article</loc></url>
    </urlset>'''
    html_path = os.path.join(os.path.dirname(__file__), "test_html", "sample_bitovi_blog.html")
    html = load_html(html_path)
    mock_requests.side_effect = [
        MagicMock(status_code=200, content=sitemap_xml.encode()),
        MagicMock(status_code=200, content=html.encode())
    ]
    result = fetch_and_parse_blogs()
    assert result["count"] == 1
    article = result["sample_articles"][0]
    assert article["title"] == "Comparing Schema Validation Libraries: AJV, Joi, Yup, and Zod"
    assert article["author"] == "Roy Ayoola"
    assert "schema validator" in article["description"]
    assert any("node.js" in t.lower() for t in article["tags"])
    assert "2023" in article["date_published"]
    mock_insert.assert_called() 