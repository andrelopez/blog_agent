import pytest
from unittest.mock import patch, MagicMock
from blog_ingest import embedding_service


def test_get_local_embedding_model():
    model = embedding_service.get_local_embedding_model()
    assert hasattr(model, 'embed')

def test_detect_date_sort_order():
    assert embedding_service.detect_date_sort_order('What are the latest posts?') == 'desc'
    assert embedding_service.detect_date_sort_order('Show me the oldest articles') == 'asc'
    assert embedding_service.detect_date_sort_order('Tell me about FastAPI') is None

@patch('blog_ingest.embedding_service.get_local_embedding_model')
@patch('blog_ingest.embedding_service.get_qdrant_client')
def test_semantic_search(mock_get_qdrant_client, mock_get_local_embedding_model):
    # Mock embedding model
    mock_model = MagicMock()
    mock_model.embed.return_value = [[0.1, 0.2, 0.3]]
    mock_get_local_embedding_model.return_value = mock_model
    # Mock Qdrant client
    mock_client = MagicMock()
    mock_hit = MagicMock()
    mock_hit.payload = {'title': 'Test', 'tags': ['ai'], 'text': 'sample', 'score': 0.99}
    mock_hit.score = 0.99
    mock_client.search.return_value = [mock_hit]
    mock_get_qdrant_client.return_value = mock_client
    results = embedding_service.semantic_search('test', top_k=1)
    assert isinstance(results, list)
    assert results[0]['title'] == 'Test'
    assert results[0]['score'] == 0.99 