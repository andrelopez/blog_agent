import logging
import requests
from apscheduler.schedulers.blocking import BlockingScheduler
from .repositories.repository import fetch_all_articles_from_db
from .services.embedding_service import embed_and_store_in_qdrant

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_URL = "http://blog_ingest:8000/ingest"  # Use service name for Docker networking


def run_ingestion_and_embedding():
    logger.info("Running Ingestion task...")
    try:
        resp = requests.post(API_URL, timeout=600)
        resp.raise_for_status()
        data = resp.json()
        logger.info(f"Ingestion complete. URLs ingested: {data.get('count')}")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        return

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


def main():
    scheduler = BlockingScheduler()
    scheduler.add_job(run_ingestion_and_embedding, 'interval', days=1)
    logger.info("Scheduler started. Will run ingestion and embedding every 24 hours.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")

if __name__ == "__main__":
    main() 