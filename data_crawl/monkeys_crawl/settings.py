# Scrapy settings for monkeys_crawl project
# ─────────────────────────────────────────────────────────────────────────────
# Documentation:
#   https://docs.scrapy.org/en/latest/topics/settings.html
#
# This file keeps **all** project-wide settings in one place.  Values that we
# tuned for large-scale distributed crawling are gathered near the bottom under
# “Networking / AutoThrottle / Logging”.

BOT_NAME = "monkeys_crawl"

SPIDER_MODULES = ["monkeys_crawl.spiders"]
NEWSPIDER_MODULE = "monkeys_crawl.spiders"

# ─────────────────────────────────────────────────────────────────────────────
# Core crawl politeness / scope
# ---------------------------------------------------------------------------
DEPTH_LIMIT = 5                  # don’t wander infinitely
ROBOTSTXT_OBEY = True            # respect robots.txt
CONCURRENT_REQUESTS = 32         # global concurrency
DOWNLOAD_DELAY = 0.2             # 200 ms between same-host requests

# ─────────────────────────────────────────────────────────────────────────────
# Pipelines
# ---------------------------------------------------------------------------
ITEM_PIPELINES = {
    "monkeys_crawl.pipelines.CleanPipeline": 300,
}

FEEDS = {
    "run_%(time)s.jl": {"format": "jsonlines", "encoding": "utf8"},
}

# ─────────────────────────────────────────────────────────────────────────────
# AutoThrottle
# ---------------------------------------------------------------------------
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 0.1       # seconds
AUTOTHROTTLE_MAX_DELAY   = 3         # seconds
AUTOTHROTTLE_TARGET_CONCURRENCY = 8.0

# ─────────────────────────────────────────────────────────────────────────────
# Networking & retry
# ---------------------------------------------------------------------------
DOWNLOAD_TIMEOUT   = 60              # fail request after 60 s
RETRY_ENABLED      = True
RETRY_TIMES        = 2               # +1 original = max 3 attempts
RETRY_HTTP_CODES   = [500, 502, 503, 504, 522, 524, 408]
DOWNLOAD_MAXSIZE   = 6 * 1024 * 1024   # 6 MB per response cap

# ─────────────────────────────────────────────────────────────────────────────
# Scrapy-Redis (distributed queue)
# ---------------------------------------------------------------------------
SCHEDULER        = "scrapy_redis.scheduler.Scheduler"
DUPEFILTER_CLASS = "scrapy_redis.dupefilter.RFPDupeFilter"
REDIS_URL        = "redis://localhost:6379"
SCHEDULER_PERSIST = True

# ─────────────────────────────────────────────────────────────────────────────
# Misc / future-proof
# ---------------------------------------------------------------------------
REQUEST_FINGERPRINTER_IMPLEMENTATION = "2.7"
TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"
FEED_EXPORT_ENCODING = "utf-8"
LOG_LEVEL = "INFO"                 # keep stack-traces out of INFO logs

# Close after 48 h
CLOSESPIDER_TIMEOUT = 172800          # 48 h in seconds
SCHEDULER_IDLE_BEFORE_CLOSE = 600     # 10-min idle window