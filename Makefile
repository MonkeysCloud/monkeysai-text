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

# ───────────────────────────────────────────────────────────────
# Convert latest JL to Parquet for training
parquet:
	mkdir -p services/text/data
	poetry run python scripts/jl_to_parquet.py \
	  $(shell ls crawl/run_*.jl | tail -1)

# ───────────────────────────────────────────────────────────────
# Train language‑model on snapshot.parquet
train:
	poetry run python -m app.train

# ───────────────────────────────────────────────────────────────
# Serve FastAPI with latest checkpoint
serve:
	CKPT=$(shell ls checkpoints/epoch*.pt | tail -1) \
	poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000
