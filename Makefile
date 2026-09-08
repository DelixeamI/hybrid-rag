.PHONY: calibrate sync up down local-pg test test-all lint fmt ingest golden-check eval-retrieval eval record demo

# fixture = synthetic mini corpus (CI, no data needed); full = real Labor Code from data/raw/
DATASET ?= fixture
RUN = uv run tkrag

sync:
	uv sync

up:
	docker compose up -d --wait

down:
	docker compose down

# Docker-free Postgres + pgvector for machines without Docker (writes DATABASE_URL to .env).
local-pg:
	uv run python scripts/local_pg.py

# Unit tests only: no Postgres, no network, no models.
test:
	uv run pytest

# Everything, including tests that need Postgres (`make up` or `make local-pg` first).
test-all:
	uv run pytest -m "db or not db"

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

fmt:
	uv run ruff check --fix .
	uv run ruff format .

ingest:
	$(RUN) ingest --dataset $(DATASET)

golden-check:
	$(RUN) golden-check --dataset $(DATASET)

# Best reranker score per golden question + suggested refusal threshold
calibrate:
	$(RUN) calibrate --dataset $(DATASET)

# Recall@k / MRR / latency per mode -> reports/results.md and .json
eval-retrieval:
	$(RUN) eval-retrieval --dataset $(DATASET)

# Retrieval + answers with RecordedLLM + CI gate (exit 1 below eval_thresholds.yaml)
eval:
	$(RUN) eval --dataset $(DATASET)

# Same eval on the REAL LLM from .env; overwrites tests/cassettes/
record:
	$(RUN) record --dataset $(DATASET)

demo: up
	$(RUN) ingest --dataset $(DATASET)
	TKRAG_DATASET=$(DATASET) uv run python scripts/demo.py
