from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import os
import uuid

QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "blog_articles")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
BATCH_SIZE = 50

def get_local_embedding_model():
    return TextEmbedding(model_name=EMBED_MODEL)

def get_qdrant_client():
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def ensure_qdrant_collection(client, vector_size):
    if QDRANT_COLLECTION not in [c.name for c in client.get_collections().collections]:
        client.recreate_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )

def embed_and_store_in_qdrant(articles):
    model = get_local_embedding_model()
    # Get embedding size from a sample embedding
    sample_embedding = list(model.embed([articles[0]["text"]]))[0]
    ensure_qdrant_collection(get_qdrant_client(), vector_size=len(sample_embedding))
    client = get_qdrant_client()
    points = []
    for i in range(0, len(articles), BATCH_SIZE):
        batch = articles[i:i+BATCH_SIZE]
        texts = [a["text"] for a in batch]
        embeddings = list(model.embed(texts))
        for article, embedding in zip(batch, embeddings):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, article["url"]))
            points.append(PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "url": article["url"],
                    "title": article["title"],
                    "date_published": str(article["date_published"]),
                    "date_modified": str(article["date_modified"]),
                    "author": article["author"],
                    "description": article["description"],
                    "text": article["text"]
                }
            ))
    client.upsert(collection_name=QDRANT_COLLECTION, points=points) 