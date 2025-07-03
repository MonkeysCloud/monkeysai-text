.PHONY: crawl-big parquet train serve

# ───────────────────────────────────────────────────────────────
crawl-big:
	# 48-hour, unlimited-item scrape
	cd data_crawl && \
	poetry run scrapy crawl discover \
	  -s JOBDIR=.check \
	  -L INFO \
	  -s CLOSESPIDER_ITEMCOUNT=0 \
	  -s CLOSESPIDER_TIMEOUT=172800 \
	  -o ../crawl/run_$(shell date +%F_%H%M).jl

# ───────────────────────────────────────────────────────────────
parquet:
	mkdir -p services/text/data
	poetry run python scripts/jl_to_parquet.py \
	  $(shell ls crawl/run_*.jl | tail -1)

train:
	poetry run python -m app.train

serve:
	CKPT=$(shell ls checkpoints/epoch*.pt | tail -1) \
	poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000