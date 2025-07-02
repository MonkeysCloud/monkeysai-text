import scrapy
from urllib.parse import urlparse

class DocsSpider(scrapy.Spider):
    name = "docs"
    allowed_domains = ["wikipedia.org", "docs.python.org"]
    start_urls = [
        "https://en.wikipedia.org/wiki/Content_management_system",
        "https://docs.python.org/3/library/asyncio.html",
    ]

    def parse(self, response):
        # emit the cleaned page text
        paragraphs = response.css("p::text").getall()
        text = " ".join(p.strip() for p in paragraphs if p.strip())
        yield {
            "url": response.url,
            "text": text,
        }
        # follow in-domain links
        for href in response.css("a::attr(href)").getall():
            abs_url = response.urljoin(href)
            dom = urlparse(abs_url).netloc
            if any(d in dom for d in self.allowed_domains):
                yield scrapy.Request(abs_url, callback=self.parse)