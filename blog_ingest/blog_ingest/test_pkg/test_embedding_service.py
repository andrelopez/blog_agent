import pytest
from unittest.mock import patch, MagicMock
from blog_ingest.services import embedding_service


def test_get_openai_embeddings():
    with patch('blog_ingest.services.embedding_service.get_openai_client') as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance
        mock_instance.embeddings.create.return_value.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        result = embedding_service.get_openai_embeddings(["test"])
        assert isinstance(result, list)
        assert result[0] == [0.1, 0.2, 0.3]

def test_detect_date_sort_order():
    assert embedding_service.detect_date_sort_order('What are the latest posts?') == 'desc'
    assert embedding_service.detect_date_sort_order('Show me the oldest articles') == 'asc'
    assert embedding_service.detect_date_sort_order('Tell me about FastAPI') is None

@patch('blog_ingest.services.embedding_service.get_openai_embeddings')
@patch('blog_ingest.services.embedding_service.get_qdrant_client')
def test_semantic_search(mock_get_qdrant_client, mock_get_openai_embeddings):
    # Mock embedding
    mock_get_openai_embeddings.return_value = [[0.1, 0.2, 0.3]]
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