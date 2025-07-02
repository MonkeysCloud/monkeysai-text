from scrapy_redis.spiders import RedisSpider
import scrapy
from urllib.parse import urlparse

class DomainDiscoverySpider(RedisSpider):
    name      = "discover"
    redis_key = "discover:start_urls"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # persistent set across requests in this process
        self.seen_domains = set()

    def parse(self, response):
        # yield the page text, etc…
        yield {
            "url": response.url,
            "text": " ".join(response.css("p::text").getall()),
        }

        for href in response.css("a::attr(href)").getall():
            abs_url = response.urljoin(href)
            dom     = urlparse(abs_url).netloc.lower()
            # if domain brand new, schedule its homepage
            if dom not in self.seen_domains:
                self.seen_domains.add(dom)
                # start with its root, or follow this URL directly
                start_url = f"https://{dom}/"
                yield scrapy.Request(start_url, callback=self.parse)