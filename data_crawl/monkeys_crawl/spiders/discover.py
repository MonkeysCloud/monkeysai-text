from scrapy_redis.spiders import RedisSpider
import scrapy, re
from urllib.parse import urljoin, urldefrag

class DiscoverSpider(RedisSpider):
    name = "discover"
    redis_key = "discover:start_urls"        # initial seeds only

    custom_settings = {
        "SCHEDULER_IDLE_BEFORE_CLOSE": 600,  # 10-min idle grace
    }

    def parse(self, response):
        # ── 1) yield the page for pipelines / storage ───────────
        paragraphs = response.css("p::text").getall()
        text = " ".join(p.strip() for p in paragraphs if p.strip())
        yield {"url": response.url, "text": text}

        # ── 2) extract EVERY link and schedule it ───────────────
        for href in response.css("a::attr(href)").getall():
            url = urldefrag(urljoin(response.url, href))[0]
            if not re.match(r"^https?://", url):
                continue
            yield scrapy.Request(url, callback=self.parse)