import openai
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from ..repositories.repository import fetch_all_articles_from_db
import os
import uuid
import tiktoken

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "blog_articles")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-3.5-turbo")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
BATCH_SIZE = 50
MAX_EMBED_TOKENS = 8192

def get_openai_client():
    return openai.OpenAI(api_key=OPENAI_API_KEY)

def get_qdrant_client():
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def get_token_encoder():
    return tiktoken.encoding_for_model(OPENAI_EMBED_MODEL)

def truncate_to_max_tokens(text, max_tokens=MAX_EMBED_TOKENS):
    encoding = get_token_encoder()
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        text = encoding.decode(tokens)
    return text

def ensure_qdrant_collection(client, vector_size):
    if QDRANT_COLLECTION not in [c.name for c in client.get_collections().collections]:
        client.recreate_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )

def get_openai_embeddings(texts):
    client = get_openai_client()
    response = client.embeddings.create(
        input=texts,
        model=OPENAI_EMBED_MODEL
    )
    return [d.embedding for d in response.data]

def run_ingestion_and_embedding():
    logger.info("Fetching articles from DB...")
    try:
        articles = fetch_all_articles_from_db()
        logger.info(f"Fetched {len(articles)} articles from DB.")
    except Exception as e:
        logger.error(f"DB fetch failed: {e}")
        return

    logger.info("Running Embedding service...")
    try:
        embed_and_store_in_qdrant(articles)
        logger.info("Embedding and storage in Qdrant successful.")
    except Exception as e:
        logger.error(f"Embedding service failed: {e}")

def embed_and_store_in_qdrant(articles):
    # Get embedding size from a sample embedding
    sample_article = articles[0]
    sample_text = truncate_to_max_tokens(
        f"{sample_article.get('title', '')}\n{sample_article.get('author', '')}\n{sample_article.get('date_published', '')}\n{sample_article.get('description', '')}\n{' '.join(sample_article.get('tags', []))}\n{sample_article['text']}"
    )
    sample_embedding = get_openai_embeddings([sample_text])[0]
    ensure_qdrant_collection(get_qdrant_client(), vector_size=len(sample_embedding))
    client = get_qdrant_client()
    points = []
    for i in range(0, len(articles), BATCH_SIZE):
        batch = articles[i:i+BATCH_SIZE]
        texts = [
            truncate_to_max_tokens(
                f"{a.get('title', '')}\n{a.get('author', '')}\n{a.get('date_published', '')}\n{a.get('description', '')}\n{' '.join(a.get('tags', []))}\n{a['text']}"
            )
            for a in batch
        ]
        embeddings = get_openai_embeddings(texts)
        for article, embedding in zip(batch, embeddings):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, article["url"]))
            points.append(PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "pageContent": article["text"],
                    "metadata": {
                        "url": article["url"],
                        "title": article["title"],
                        "date_published": str(article["date_published"]),
                        "date_modified": str(article["date_modified"]),
                        "author": article["author"],
                        "description": article["description"],
                        "tags": article.get("tags", [])
                    }
                }
            ))
    client.upsert(collection_name=QDRANT_COLLECTION, points=points) 