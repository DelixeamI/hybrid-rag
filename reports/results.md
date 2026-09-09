# Evaluation results

Dataset: **fixture** — 20 articles, 20 chunks, 20 questions (`golden_fixture.jsonl`).

> The `fixture` corpus is SYNTHETIC: a few dozen hand-written approximations of articles. These numbers check that the pipeline works; they say nothing about quality on the real Labor Code. Small n: one question moves a percentage by several points.

Embeddings `intfloat/multilingual-e5-small`, reranker `BAAI/bge-reranker-v2-m3`, RRF k=10 (dense weight 3), rerank top-20 -> top-5. Recall@k: a question counts as found when ANY expected article is among the top-k distinct articles (alternatives); for `multi_article` it is the SHARE of the required articles found, and MRR uses the rank where the last required article appears.

## Retrieval

| Mode | Recall@1 | Recall@5 | Recall@10 | MRR | latency p50, ms | latency p95, ms |
|---|---|---|---|---|---|---|
| Postgres FTS (ts_rank_cd) | 28.1% | 78.1% | 78.1% | 0.390 | 0.9 | 3.0 |
| BM25 (Postgres lexemes) | 50.0% | 78.1% | 81.2% | 0.567 | 0.6 | 0.8 |
| dense (e5) | 84.4% | 96.9% | 100% | 0.844 | 12.2 | 90.0 |
| hybrid (dense + BM25, RRF) | 78.1% | 96.9% | 100% | 0.802 | 12.4 | 16.4 |
| hybrid + rerank | 90.6% | 100% | 100% | 0.896 | 186.1 | 202.8 |

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

| Metric | Value | n |
|---|---|---|
| Correct refusals on `no_answer` | 100% | 4 |
| False refusals on answerable questions | 0% | 16 |
| `answer_facts` recall (answerable, refusals count as 0; prepositions ignored) | 100% | 16 |
| `answer_facts` recall, strict (also demands prepositions and conjunctions) | 100% | 16 |
| Grounded answers: numbers, articles and quotes are in the cited text (share of the answers given, not of all questions) | 100% | 16 |

Outcome statuses: ok=16, refused=4

**How to read these numbers.**

- *Grounded answers* is a syntactic check: the numbers with units, the article numbers and the quotes of the answer occur in the cited fragments. It is not the share of legally correct answers; a grounded answer can still draw a wrong conclusion from the right article. It also counts only the answers that were given.
- *`answer_facts` recall* is lexical overlap with the expected facts (words and numbers after a crude stemming). It misses a correct answer in other words and can accept a wrong answer that contains the same words.
- *Correct refusals* depend on a threshold that was chosen on this same golden set, so they are optimistic for new questions. Only a set that was not used to choose any parameter or threshold gives an honest estimate (see `eval --frozen` in the README).

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
