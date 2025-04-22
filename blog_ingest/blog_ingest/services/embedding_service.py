import openai
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import os
import uuid
import re
from datetime import datetime
import tiktoken

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

def hybrid_search(question, top_k=20):
    """
    If the question is about latest/oldest, return articles sorted by date. Otherwise, do semantic search.
    """
    order = detect_date_sort_order(question)
    if order:
        articles = get_all_articles_sorted_by_date(order=order)
        return articles[:top_k]
    else:
        return semantic_search(question, top_k=top_k)

def semantic_search(question, top_k=20):
    """
    Embed the question, search Qdrant for top_k most similar articles, and return their payloads.
    """
    client = get_qdrant_client()
    question_embedding = get_openai_embeddings([
        truncate_to_max_tokens(question)
    ])[0]
    search_result = client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=question_embedding,
        limit=top_k,
        with_payload=True
    )
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
        context += (
            f"[{idx}] Title: {a.get('title')}\n"
            f"Author: {a.get('author')}\n"
            f"Date: {a.get('date_published')}\n"
            f"Tags: {', '.join(a.get('tags', []))}\n"
            f"URL: {a.get('url')}\n"
            f"Content: {a.get('text')[:1000]}...\n\n"
        )
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