# blog_ingest

This service is responsible for scraping Bitovi blog articles, extracting clean text and metadata, and storing them in Postgres for downstream RAG workflows.

## Features
- Parse Bitovi's sitemap to extract all /blog/ URLs
- Fetch and clean each blog post (remove HTML)
- Collect metadata (title, date, tags, URL)
- Store results in Postgres
- Implement polite scraping (200ms delay, error handling, retries)
- Allow scheduling via environment variable

## Setup
Instructions will be added as the service is implemented. 