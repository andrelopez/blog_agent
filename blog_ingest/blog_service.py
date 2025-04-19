import requests
import xml.etree.ElementTree as ET
import time
from bs4 import BeautifulSoup
import json
from dateutil import parser as dateparser
import logging
from repository import get_pg_connection, ensure_blog_articles_table, insert_blog_article

SITEMAP_URL = "https://www.bitovi.com/sitemap.xml"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def normalize_date(raw_date):
    if not raw_date:
        return None
    try:
        return dateparser.parse(raw_date).isoformat()
    except Exception:
        return raw_date

def fetch_and_parse_blogs():
    conn = get_pg_connection()
    ensure_blog_articles_table(conn)
    resp = requests.get(SITEMAP_URL, timeout=10)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    ns = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
    blog_urls = []
    for url in root.findall('ns:url', ns):
        loc = url.find('ns:loc', ns)
        if loc is not None and '/blog/' in loc.text:
            blog_urls.append(loc.text)
    articles = []
    for url in blog_urls:
        try:
            page = requests.get(url, timeout=10)
            page.raise_for_status()
            soup = BeautifulSoup(page.content, 'html.parser')
            # Extract title
            title = soup.title.string.strip() if soup.title else None
            # Extract JSON-LD metadata (find the script with datePublished)
            date_published = date_modified = author = description = None
            json_ld_tags = soup.find_all('script', type='application/ld+json')
            for json_ld in json_ld_tags:
                try:
                    data = json.loads(json_ld.string)
                    # Only use JSON-LD blocks that are BlogPosting
                    if isinstance(data, dict) and data.get('@type') == 'BlogPosting':
                        date_published = normalize_date(data.get('datePublished'))
                        date_modified = normalize_date(data.get('dateModified'))
                        author = data.get('author', {}).get('name') if isinstance(data.get('author'), dict) else None
                        description = data.get('description')
                        break
                except Exception:
                    continue
            # Fallback: Published date from <time>
            if not date_published:
                time_tag = soup.find('time')
                if time_tag and time_tag.text.strip():
                    date_published = normalize_date(time_tag.text.strip())

            # Fallback: Author from <p class="text-[16px] font-bold">
            if not author:
                author_tag = soup.find('p', class_='text-[16px] font-bold')
                if author_tag and author_tag.text.strip():
                    author = author_tag.text.strip()
            # Fallback: Description from <meta name="description">
            if not description:
                desc_tag = soup.find('meta', attrs={'name': 'description'})
                if desc_tag and desc_tag.get('content'):
                    description = desc_tag['content']
              # we are unable to get date_published, let's skip this article:
            
            if not date_published:
                logger.error(f"Skipping {url}: missing metadata")
                continue
            # Extract main text
            main = soup.find('article') or soup.find('main') or soup.body
            text = main.get_text(separator=' ', strip=True) if main else ''
            article = {
                'url': url,
                'title': title,
                'date_published': date_published,
                'date_modified': date_modified,
                'author': author,
                'description': description,
                'text': text,
                'text_length': len(text),
                'text_preview': text[:200]
            }
            articles.append(article)
            insert_blog_article(conn, article)
            time.sleep(0.2)
        except Exception as e:
            logger.error(f"Error processing {url}: {e}")
            continue
    conn.close()
    return {
        "count": len(blog_urls),
        "sample_articles": articles,
    } 