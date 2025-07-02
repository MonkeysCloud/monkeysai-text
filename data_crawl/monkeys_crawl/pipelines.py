import hashlib, datetime, pytz, scrapy, re
import trafilatura, requests
from readability import Document
from bs4 import BeautifulSoup

MIN_TOKENS = 50          # drop short blurbs

def _extract_clean(url: str) -> dict:
    html = requests.get(url, timeout=15).text
    text = trafilatura.extract(html, include_comments=False, include_tables=False)
    if not text:                                   # fallback
        soup = BeautifulSoup(Document(html).summary(), "lxml")
        text = soup.get_text(" ", strip=True)
    return {
        "title": Document(html).short_title(),
        "text": re.sub(r"\\s+", " ", text).strip()
    }

class CleanPipeline:
    def process_item(self, item, spider):
        url = item["url"]
        try:
            art = _extract_clean(url)
            if len(art["text"].split()) < MIN_TOKENS:
                raise scrapy.exceptions.DropItem()

            doc_id = hashlib.sha1(url.encode()).hexdigest()
            return {
                "id": doc_id,
                "url": url,
                "title": art["title"],
                "crawl_date": datetime.datetime.now(tz=pytz.UTC).isoformat(),
                "text": art["text"],
            }
        except Exception as e:
            spider.logger.debug(f"skip {url}: {e}")
            raise scrapy.exceptions.DropItem()