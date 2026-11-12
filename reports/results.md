# Evaluation results

Dataset: **fixture** — 20 articles, 20 chunks, 20 questions (`golden_fixture.jsonl`).

> The `fixture` corpus is SYNTHETIC: a few dozen hand-written approximations of articles. These numbers check that the pipeline works; they say nothing about quality on the real Labor Code. Small n: one question moves a percentage by several points.

Embeddings `intfloat/multilingual-e5-small`, reranker `BAAI/bge-reranker-v2-m3`, RRF k=10 (dense weight 3), rerank top-20 -> top-5. Recall@k: a question counts as found when ANY expected article is among the top-k distinct articles (alternatives); for `multi_article` it is the SHARE of the required articles found, and MRR uses the rank where the last required article appears.

## Retrieval

| Mode | Recall@1 | Recall@5 | Recall@10 | MRR | latency p50, ms | latency p95, ms |
|---|---|---|---|---|---|---|
| Postgres FTS (ts_rank_cd) | 28.1% | 78.1% | 78.1% | 0.390 | 2.0 | 3.5 |
| BM25 (Postgres lexemes) | 50.0% | 78.1% | 81.2% | 0.567 | 0.6 | 0.8 |
| dense (e5) | 84.4% | 96.9% | 100% | 0.844 | 16.8 | 84.8 |
| hybrid (dense + BM25, RRF) | 78.1% | 96.9% | 100% | 0.802 | 20.1 | 25.5 |
| hybrid + rerank | 90.6% | 100% | 100% | 0.896 | 331.6 | 3460.4 |

**Recall@1 by question type**

| Type | n | Postgres FTS (ts_rank_cd) | BM25 (Postgres lexemes) | dense (e5) | hybrid (dense + BM25, RRF) | hybrid + rerank |
|---|---|---|---|---|---|---|
| factual | 7 | 42.9% | 100% | 100% | 100% | 100% |
| multi_article | 3 | 50.0% | 33.3% | 50.0% | 50.0% | 50.0% |
| paraphrase | 6 | 0% | 0% | 83.3% | 66.7% | 100% |

**Recall@5 by question type**

| Type | n | Postgres FTS (ts_rank_cd) | BM25 (Postgres lexemes) | dense (e5) | hybrid (dense + BM25, RRF) | hybrid + rerank |
|---|---|---|---|---|---|---|
| factual | 7 | 100% | 100% | 100% | 100% | 100% |
| multi_article | 3 | 83.3% | 83.3% | 83.3% | 83.3% | 100% |
| paraphrase | 6 | 50.0% | 50.0% | 100% | 100% | 100% |

**MRR by question type**

| Type | n | Postgres FTS (ts_rank_cd) | BM25 (Postgres lexemes) | dense (e5) | hybrid (dense + BM25, RRF) | hybrid + rerank |
|---|---|---|---|---|---|---|
| factual | 7 | 0.714 | 1.000 | 1.000 | 1.000 | 1.000 |
| multi_article | 3 | 0.178 | 0.389 | 0.389 | 0.389 | 0.444 |
| paraphrase | 6 | 0.117 | 0.150 | 0.889 | 0.778 | 1.000 |

### Misses

Questions not fully found in the top 5 of `hybrid_rerank` (nothing found, or for `multi_article` at least one required article missing): 0 of 16.

## Answers

LLM: `gemini-3.8-flash`; refusal threshold (reranker score): 0.003

| Metric | All | DEV (threshold fitted here) | TEST (held out) |
|---|---|---|---|
| Correct refusals on `no_answer` | 4/4 (100%) | 2/2 (100%) | 2/2 (100%) |
| False refusals on answerable questions | 0/16 (0%) | 0/9 (0%) | 0/7 (0%) |
| `answer_facts` recall (refusals count as 0; prepositions ignored) | 100% | 100% | 100% |
| `answer_facts` recall, strict (also demands prepositions and conjunctions) | 100% | 100% | 100% |
| Grounded answers: numbers, articles and quotes are in the cited text (of the answers given, not of all questions) | 16/16 (100%) | 9/9 (100%) | 7/7 (100%) |

Outcome statuses (all): ok=16, refused=4

**How to read these numbers.**

- *Grounded answers* is a syntactic check: the numbers with units, the article numbers and the quotes of the answer occur in the cited fragments. It is not the share of legally correct answers; a grounded answer can still draw a wrong conclusion from the right article. It also counts only the answers that were given.
- *`answer_facts` recall* is lexical overlap with the expected facts (words and numbers after a crude stemming). It misses a correct answer in other words and can accept a wrong answer that contains the same words.
- *Correct refusals*: the threshold was fitted on the DEV half only, so the TEST column is a held-out estimate. TEST holds few `no_answer` questions, so it is a rough one.
- The retrieval parameters (RRF k and dense weight) were chosen on the DEV half of the answerable questions (`reports/experiment_hybrid.md`); the TEST half was not used.

## Refusal threshold

Refuse when the best reranker score is below the threshold. The threshold with the best (correct refusals - false refusals) on the DEV half (11 questions) is **0.1817**. The TEST half was not used to choose it.

| Threshold | Correct refusals | False refusals |
|---|---|---|
| 0.0 | 0% | 0% |
| 0.0001 | 50.0% | 0% |
| 0.1817 | 100% | 0% |
| 0.4652 | 100% | 11.1% |
| 0.5974 | 100% | 22.2% |
| 0.7467 | 100% | 33.3% |
| 0.9184 | 100% | 44.4% |
| 0.9783 | 100% | 55.6% |
| 0.9881 | 100% | 66.7% |
| 0.9949 | 100% | 77.8% |
| 0.9991 | 100% | 88.9% |
| 0.9996 | 100% | 100% |

## CI gate

| Check | Value | Threshold | Result |
|---|---|---|---|
| recall_at_5 | 100% | >= 90.0% | pass |
| correct_refusal_rate | 100% | >= 75.0% | pass |
