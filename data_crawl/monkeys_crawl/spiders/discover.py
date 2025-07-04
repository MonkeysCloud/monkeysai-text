# data_crawl/monkeys_crawl/spiders/discover.py
from scrapy_redis.spiders import RedisSpider
import scrapy
from urllib.parse import urljoin, urldefrag

class DiscoverSpider(RedisSpider):
    name = "discover"
    redis_key = "discover:start_urls"          # seeds pushed here

    custom_settings = {
        # let Scrapy wait for new URLs before closing
        "SCHEDULER_IDLE_BEFORE_CLOSE": 600,    # 10 min grace
    }

    def parse(self, response):
        # ① extract and clean the page text — pipeline will drop short/dupes
        paragraphs = response.css("p::text").getall()
        text = " ".join(p.strip() for p in paragraphs if p.strip())
        yield {"url": response.url, "text": text}

        # ② push EVERY <a> link back to the scheduler
        for href in response.css("a::attr(href)").getall():
            abs_url = urldefrag(urljoin(response.url, href))[0]
            # rely on dupefilter & depth_limit to avoid loops
            yield scrapy.Request(abs_url, callback=self.parse, dont_filter=False)