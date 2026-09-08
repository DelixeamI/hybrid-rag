# Evaluation results

Dataset: **fixture** — 20 articles, 20 chunks, 20 questions (`golden_fixture.jsonl`).

> The `fixture` corpus is SYNTHETIC: a few dozen hand-written approximations of articles. These numbers check that the pipeline works; they say nothing about quality on the real Labor Code. Small n: one question moves a percentage by several points.

Embeddings `intfloat/multilingual-e5-small`, reranker `BAAI/bge-reranker-v2-m3`, RRF k=60 (dense weight 3), rerank top-20 -> top-5. Hit = at least one expected article among the top-k distinct articles.

## Retrieval

| Mode | Recall@1 | Recall@5 | Recall@10 | MRR | latency p50, ms | latency p95, ms |
|---|---|---|---|---|---|---|
| Postgres FTS (ts_rank_cd) | 37.5% | 81.2% | 81.2% | 0.544 | 2.3 | 3.8 |
| BM25 (Postgres lexemes) | 56.2% | 81.2% | 81.2% | 0.640 | 1.1 | 1.8 |
| dense (e5) | 93.8% | 100% | 100% | 0.958 | 13.8 | 69.5 |
| hybrid (dense + BM25, RRF) | 81.2% | 87.5% | 100% | 0.850 | 14.8 | 25.7 |
| hybrid + rerank | 100% | 100% | 100% | 1.000 | 193.1 | 207.8 |

**Recall@1 by question type**

| Type | n | Postgres FTS (ts_rank_cd) | BM25 (Postgres lexemes) | dense (e5) | hybrid (dense + BM25, RRF) | hybrid + rerank |
|---|---|---|---|---|---|---|
| factual | 7 | 42.9% | 100% | 100% | 100% | 100% |
| multi_article | 3 | 100% | 66.7% | 100% | 100% | 100% |
| paraphrase | 6 | 0% | 0% | 83.3% | 50.0% | 100% |

**Recall@5 by question type**

| Type | n | Postgres FTS (ts_rank_cd) | BM25 (Postgres lexemes) | dense (e5) | hybrid (dense + BM25, RRF) | hybrid + rerank |
|---|---|---|---|---|---|---|
| factual | 7 | 100% | 100% | 100% | 100% | 100% |
| multi_article | 3 | 100% | 100% | 100% | 100% | 100% |
| paraphrase | 6 | 50.0% | 50.0% | 100% | 66.7% | 100% |

**MRR by question type**

| Type | n | Postgres FTS (ts_rank_cd) | BM25 (Postgres lexemes) | dense (e5) | hybrid (dense + BM25, RRF) | hybrid + rerank |
|---|---|---|---|---|---|---|
| factual | 7 | 0.714 | 1.000 | 1.000 | 1.000 | 1.000 |
| multi_article | 3 | 1.000 | 0.778 | 1.000 | 1.000 | 1.000 |
| paraphrase | 6 | 0.117 | 0.150 | 0.889 | 0.600 | 1.000 |

### Misses

Questions with no expected article in the top 5 of `hybrid_rerank`: 0 of 16.

## Answers

LLM: `gemini-3.8-flash`; refusal threshold (reranker score): 0.003

| Metric | Value | n |
|---|---|---|
| Correct refusals on `no_answer` | 100% | 4 |
| False refusals on answerable questions | 0% | 16 |
| `answer_facts` recall (answerable, refusals count as 0; prepositions ignored) | 100% | 16 |
| `answer_facts` recall, strict (also demands prepositions and conjunctions) | 100% | 16 |
| Answers with every number and article confirmed by citations | 100% | 16 |

Outcome statuses: ok=16, refused=4

## Refusal threshold

Refuse when the best reranker score is below the threshold. The threshold with the best (correct refusals - false refusals) is **0.0032**. It is a suggestion fitted on this same golden set, not an independent estimate.

| Threshold | Correct refusals | False refusals |
|---|---|---|
| 0.0 | 0% | 0% |
| 0.0001 | 25.0% | 0% |
| 0.0009 | 75.0% | 0% |
| 0.0032 | 100% | 0% |
| 0.0248 | 100% | 6.2% |
| 0.1855 | 100% | 12.5% |
| 0.3448 | 100% | 18.8% |
| 0.4652 | 100% | 25.0% |
| 0.5974 | 100% | 31.2% |
| 0.6779 | 100% | 37.5% |
| 0.7968 | 100% | 43.8% |
| 0.9184 | 100% | 50.0% |
| 0.9783 | 100% | 56.2% |
| 0.9881 | 100% | 62.5% |
| 0.9932 | 100% | 68.8% |
| 0.9961 | 100% | 75.0% |
| 0.9978 | 100% | 81.2% |
| 0.9989 | 100% | 87.5% |
| 0.9993 | 100% | 93.8% |
| 0.9996 | 100% | 100% |

## CI gate

| Check | Value | Threshold | Result |
|---|---|---|---|
| recall_at_5 | 100% | >= 90.0% | pass |
| correct_refusal_rate | 100% | >= 75.0% | pass |
