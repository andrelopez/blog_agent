from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import os
import uuid
import openai
import re
from datetime import datetime

QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "blog_articles")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
BATCH_SIZE = 50
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-3.5-turbo")

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

def detect_date_sort_order(question):
    """
    Returns 'desc' for latest/newest/most recent, 'asc' for oldest/first, or None for normal semantic search.
    """
    if re.search(r'latest|newest|most recent', question, re.IGNORECASE):
        return 'desc'
    if re.search(r'oldest|first', question, re.IGNORECASE):
        return 'asc'
    return None

def get_all_articles_sorted_by_date(order='desc'):
    """
    Fetch all articles from Qdrant and sort by date_published (order: 'asc' or 'desc').
    """
    client = get_qdrant_client()
    # Scroll all points (no vector search)
    points, _ = client.scroll(collection_name=QDRANT_COLLECTION, with_payload=True, limit=10000)
    def parse_date(article):
        d = article.get('date_published')
        try:
            return datetime.fromisoformat(d) if d else datetime.min
        except Exception:
            return datetime.min
    articles = [p.payload for p in points]
    articles = [a for a in articles if a.get('date_published')]
    articles.sort(key=parse_date, reverse=(order=='desc'))
    return articles

def hybrid_search(question, top_k=3):
    """
    If the question is about latest/oldest, return articles sorted by date. Otherwise, do semantic search.
    """
    order = detect_date_sort_order(question)
    if order:
        articles = get_all_articles_sorted_by_date(order=order)
        return articles[:top_k]
    else:
        return semantic_search(question, top_k=top_k)

def semantic_search(question, top_k=3):
    """
    Embed the question, search Qdrant for top_k most similar articles, and return their payloads.
    """
    model = get_local_embedding_model()
    client = get_qdrant_client()
    # Embed the question
    question_embedding = list(model.embed([question]))[0]
    # Search Qdrant
    search_result = client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=question_embedding,
        limit=top_k,
        with_payload=True
    )
    # Extract payloads (article metadata and text)
    articles = []
    for hit in search_result:
        payload = hit.payload
        payload["score"] = hit.score
        articles.append(payload)
    return articles

def generate_rag_answer(question, articles):
    """
    Call OpenAI LLM with the question and context from articles.
    Returns the generated answer.
    """
    # Build the context string from the top articles
    context = ""
    for idx, a in enumerate(articles, 1):
        context += f"[{idx}] Title: {a.get('title')}\nURL: {a.get('url')}\nContent: {a.get('text')[:500]}...\n\n"
    # Build the prompt
    prompt = (
        f"You are an expert assistant. Use the following blog articles as context to answer the user's question. "
        f"Reference the articles by their [number] and include the URL in your answer when relevant.\n\n"
        f"Context:\n{context}\n"
        f"User question: {question}\n\n"
        f"Answer (be concise, relevant, and include reference links):"
    )
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions using provided blog articles as context."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=400,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip() 