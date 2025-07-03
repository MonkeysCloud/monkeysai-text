import datetime
import hashlib
import re
from typing import Set

import pytz
import requests
import scrapy
import trafilatura
from bs4 import BeautifulSoup
from readability import Document

# ────────────────────────────────────────────────────────────────
# Config
# ----------------------------------------------------------------
MIN_WORDS: int = 300   # discard pages shorter than this
MIN_TOKENS: int = 50   # extraction-level guard (trafilatura fallback)

# In-memory duplicate filter (hash of first 256 chars)
SEEN_HASHES: Set[str] = set()

# ────────────────────────────────────────────────────────────────
# Helpers
# ----------------------------------------------------------------

def _extract_clean(url: str) -> dict:
    """Download *url*, run readability → return title/text stripped of HTML."""
    html = requests.get(url, timeout=15).text

    # First try trafilatura extractor
    text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
    )

    # Fallback to readability → BeautifulSoup if trafilatura fails
    if not text:
        soup = BeautifulSoup(Document(html).summary(), "lxml")
        text = soup.get_text(" ", strip=True)

    clean_text = re.sub(r"\s+", " ", text).strip()
    return {
        "title": Document(html).short_title(),
        "text": clean_text,
    }

# ────────────────────────────────────────────────────────────────
# Pipeline
# ----------------------------------------------------------------

class CleanPipeline:
    """Scrapy item-pipeline that:

    1. Downloads and cleans the article body (trafilatura → readability).
    2. Drops items that are too short or duplicate.
    3. Emits a normalised dict: {id, url, title, crawl_date, text}.
    """

    def process_item(self, item: dict, spider: scrapy.Spider):  # type: ignore[override]
        url: str = item.get("url", "")
        try:
            art = _extract_clean(url)

            # Token/word length guards -----------------------------------
            if len(art["text"].split()) < MIN_WORDS:
                raise scrapy.exceptions.DropItem("too short")
            if len(art["text"].split()) < MIN_TOKENS:
                raise scrapy.exceptions.DropItem("too few tokens after extract")

            # De-duplication ---------------------------------------------
            h = hashlib.md5(art["text"][:256].encode()).hexdigest()
            if h in SEEN_HASHES:
                raise scrapy.exceptions.DropItem("duplicate")
            SEEN_HASHES.add(h)

            # Return canonical record ------------------------------------
            return {
                "id": hashlib.sha1(url.encode()).hexdigest(),
                "url": url,
                "title": art["title"],
                "crawl_date": datetime.datetime.now(tz=pytz.UTC).isoformat(),
                "text": art["text"],
            }

        except Exception as exc:
            spider.logger.debug(f"skip {url}: {exc}")
            raise scrapy.exceptions.DropItem()