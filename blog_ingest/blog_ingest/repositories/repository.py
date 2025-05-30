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
        # Index for date sorting/filtering
        cur.execute('''
            CREATE INDEX IF NOT EXISTS idx_blog_articles_date_published
            ON blog_articles(date_published DESC);
        ''')
        # Index for tag searches
        cur.execute('''
            CREATE INDEX IF NOT EXISTS idx_blog_articles_tags
            ON blog_articles USING GIN (tags);
        ''')
        # Full-text search indexes
        cur.execute('''
            CREATE INDEX IF NOT EXISTS idx_blog_articles_text_search
            ON blog_articles USING GIN (to_tsvector('english', text));
        ''')
        cur.execute('''
            CREATE INDEX IF NOT EXISTS idx_blog_articles_title_search
            ON blog_articles USING GIN (to_tsvector('english', title));
        ''')
        # Index for author queries
        cur.execute('''
            CREATE INDEX IF NOT EXISTS idx_blog_articles_author
            ON blog_articles(author);
        ''')
        # Composite index for author+date queries
        cur.execute('''
            CREATE INDEX IF NOT EXISTS idx_blog_articles_author_date
            ON blog_articles(author, date_published DESC);
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

def fetch_all_articles_from_db():
    articles = []
    try:
        conn = get_pg_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT url, title, date_published, date_modified, author, description, text FROM blog_articles ORDER BY date_published DESC")
            rows = cur.fetchall()
            articles = [
                {
                    "url": row[0],
                    "title": row[1],
                    "date_published": row[2],
                    "date_modified": row[3],
                    "author": row[4],
                    "description": row[5],
                    "text": row[6],
                }
                for row in rows
            ]
        conn.close()
    except Exception as e:
        # Optionally log or raise
        raise
    return articles 