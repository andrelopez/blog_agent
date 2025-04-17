import pytest
from blog_ingest.blog_service import normalize_date, fetch_and_parse_blogs
from unittest.mock import patch, MagicMock
import datetime

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

# Test fetch_and_parse_blogs with mocks
@patch("blog_ingest.blog_service.requests.get")
@patch("blog_ingest.blog_service.get_pg_connection")
@patch("blog_ingest.blog_service.ensure_blog_articles_table")
@patch("blog_ingest.blog_service.insert_blog_article")
def test_fetch_and_parse_blogs(mock_insert, mock_ensure, mock_conn, mock_requests):
    # Mock sitemap response
    sitemap_xml = '''<?xml version=\"1.0\" encoding=\"UTF-8\"?>
    <urlset xmlns=\"http://www.sitemaps.org/schemas/sitemap/0.9\">
      <url><loc>https://www.bitovi.com/blog/test-article</loc></url>
    </urlset>'''
    mock_requests.side_effect = [
        MagicMock(status_code=200, content=sitemap_xml.encode()),
        MagicMock(status_code=200, content=b"<html><head><title>Test</title></head><body><article>Test content</article><script type='application/ld+json'>{\"datePublished\": \"2024-03-07 14:00:00\", \"author\": {\"name\": \"Author\"}, \"description\": \"desc\"}</script></body></html>")
    ]
    result = fetch_and_parse_blogs()
    assert result["count"] == 1
    assert len(result["sample_articles"]) == 1
    article = result["sample_articles"][0]
    assert article["title"] == "Test"
    assert article["author"] == "Author"
    assert article["description"] == "desc"
    assert "Test content" in article["text_preview"]
    mock_insert.assert_called() 