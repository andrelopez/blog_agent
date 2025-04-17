import requests
import xml.etree.ElementTree as ET
import time
from bs4 import BeautifulSoup
import json
from dateutil import parser as dateparser

SITEMAP_URL = "https://www.bitovi.com/sitemap.xml"

def normalize_date(raw_date):
    if not raw_date:
        return None
    try:
        return dateparser.parse(raw_date).isoformat()
    except Exception:
        return raw_date

def fetch_and_parse_blogs():
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
                    if isinstance(data, dict) and 'datePublished' in data:
                        date_published = normalize_date(data.get('datePublished'))
                        date_modified = normalize_date(data.get('dateModified'))
                        author = data.get('author', {}).get('name') if isinstance(data.get('author'), dict) else None
                        description = data.get('description')
                        break
                except Exception:
                    continue
            # Extract main text
            main = soup.find('article') or soup.find('main') or soup.body
            text = main.get_text(separator=' ', strip=True) if main else ''
            articles.append({
                'url': url,
                'title': title,
                'date_published': date_published,
                'date_modified': date_modified,
                'author': author,
                'description': description,
                'text_length': len(text),
                'text_preview': text[:200]
            })
            time.sleep(0.2)
            if len(articles) >= 3:
                break
        except Exception as e:
            articles.append({'url': url, 'error': str(e)})
    return {
        "count": len(blog_urls),
        "sample_articles": articles,
    } 