.PHONY: crawl-big parquet train serve

# ───────────────────────────────────────────────────────────────
# 48‑hour distributed crawl with tuned networking / retries

crawl-big:
	cd data_crawl && \
	poetry run scrapy crawl discover \
	  -s DOWNLOAD_TIMEOUT=60 \
	  -s RETRY_TIMES=2 \
	  -s JOBDIR=.check \
	  -L INFO \
	  -s CLOSESPIDER_TIMEOUT=172800 \
	  -s SCHEDULER_IDLE_BEFORE_CLOSE=600 \
	  -o ../crawl/run_$(shell date +%F_%H%M).jl

seed:
	./scripts/seed_discover.sh

reset-crawl:
	rm -rf data_crawl/.check
	redis-cli DEL discover:dupefilter discover:requests discover:start_urls
	./scripts/seed_discover.sh
	make crawl-big

# ───────────────────────────────────────────────────────────────
# Convert latest JL to Parquet for training
parquet:
	mkdir -p services/text/data
	@latest="$$(ls crawl/run_*.jl | sort | tail -1)" ; \
	$if [ -z "$$latest" ] ; then \
      echo "❌ no crawl/run_*.jl files found" >&2 ; exit 1 ; \
    fi ; \
    echo "⏳ converting $$latest → Parquet…" ; \
    poetry run python scripts/jl_to_parquet.py "$$latest"

# ───────────────────────────────────────────────────────────────
# Train language‑model on snapshot.parquet
train:
	poetry run python -m app.train

# ───────────────────────────────────────────────────────────────
# Serve FastAPI with latest checkpoint
serve:
	CKPT=$(shell ls checkpoints/epoch*.pt | tail -1) \
	poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000
