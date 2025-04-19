import psycopg2
import os

def get_pg_connection():
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "pg-n8n")
    POSTGRES_DB = os.getenv("POSTGRES_DB", "n8n")
    POSTGRES_USER = os.getenv("POSTGRES_USER", "n8n")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
    POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", 5432))
    return psycopg2.connect(
        host=POSTGRES_HOST,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        port=POSTGRES_PORT
    )

def ensure_blog_articles_table(conn):
    with conn.cursor() as cur:
        cur.execute('''
            CREATE TABLE IF NOT EXISTS blog_articles (
                id SERIAL PRIMARY KEY,
                url TEXT UNIQUE NOT NULL,
                title TEXT,
                date_published TIMESTAMP,
                date_modified TIMESTAMP,
                author TEXT,
                description TEXT,
                tags TEXT[],
                text TEXT
            );
        ''')
        conn.commit()

def insert_blog_article(conn, article):
    with conn.cursor() as cur:
        cur.execute('''
            INSERT INTO blog_articles (url, title, date_published, date_modified, author, description, tags, text)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (url) DO UPDATE SET
                title = EXCLUDED.title,
                date_published = EXCLUDED.date_published,
                date_modified = EXCLUDED.date_modified,
                author = EXCLUDED.author,
                description = EXCLUDED.description,
                tags = EXCLUDED.tags,
                text = EXCLUDED.text;
        ''', (
            article['url'],
            article['title'],
            article['date_published'],
            article['date_modified'],
            article['author'],
            article['description'],
            article.get('tags', []),
            article['text']
        ))
        conn.commit() 