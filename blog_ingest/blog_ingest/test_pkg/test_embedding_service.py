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