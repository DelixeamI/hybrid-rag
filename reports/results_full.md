# Evaluation results

Dataset: **full** — 534 articles, 710 chunks, 92 questions (`golden.jsonl`).

Embeddings `intfloat/multilingual-e5-small`, reranker `BAAI/bge-reranker-v2-m3`, RRF k=60 (dense weight 3), rerank top-20 -> top-5. Hit = at least one expected article among the top-k distinct articles.

## Retrieval

| Mode | Recall@1 | Recall@5 | Recall@10 | MRR | latency p50, ms | latency p95, ms |
|---|---|---|---|---|---|---|
| Postgres FTS (ts_rank_cd) | 21.1% | 42.1% | 50.0% | 0.287 | 9.2 | 122.1 |
| BM25 (Postgres lexemes) | 50.0% | 80.3% | 84.2% | 0.622 | 3.0 | 4.2 |
| dense (e5) | 75.0% | 84.2% | 92.1% | 0.795 | 20.5 | 27.4 |
| hybrid (dense + BM25, RRF) | 73.7% | 90.8% | 90.8% | 0.799 | 23.4 | 29.6 |
| hybrid + rerank | 86.8% | 93.4% | 94.7% | 0.898 | 404.1 | 466.3 |

**Recall@1 by question type**

| Type | n | Postgres FTS (ts_rank_cd) | BM25 (Postgres lexemes) | dense (e5) | hybrid (dense + BM25, RRF) | hybrid + rerank |
|---|---|---|---|---|---|---|
| factual | 33 | 33.3% | 72.7% | 93.9% | 93.9% | 97.0% |
| multi_article | 10 | 20.0% | 60.0% | 90.0% | 80.0% | 80.0% |
| paraphrase | 33 | 9.1% | 24.2% | 51.5% | 51.5% | 78.8% |

**Recall@5 by question type**

| Type | n | Postgres FTS (ts_rank_cd) | BM25 (Postgres lexemes) | dense (e5) | hybrid (dense + BM25, RRF) | hybrid + rerank |
|---|---|---|---|---|---|---|
| factual | 33 | 60.6% | 97.0% | 97.0% | 97.0% | 97.0% |
| multi_article | 10 | 40.0% | 90.0% | 100% | 100% | 100% |
| paraphrase | 33 | 24.2% | 60.6% | 66.7% | 81.8% | 87.9% |

**MRR by question type**

| Type | n | Postgres FTS (ts_rank_cd) | BM25 (Postgres lexemes) | dense (e5) | hybrid (dense + BM25, RRF) | hybrid + rerank |
|---|---|---|---|---|---|---|
| factual | 33 | 0.432 | 0.827 | 0.955 | 0.955 | 0.970 |
| multi_article | 10 | 0.271 | 0.725 | 0.933 | 0.875 | 0.875 |
| paraphrase | 33 | 0.147 | 0.385 | 0.593 | 0.621 | 0.833 |

### Misses

Questions with no expected article in the top 5 of `hybrid_rerank`: 5 of 76.

| id | type | question | expected | found (top 5) |
|---|---|---|---|---|
| d006 | paraphrase | Могут ли городские власти или чиновники министерств представлять руководство бюджетных учр | 34 | 6, 407, 33, 412, 281.1 |
| d020 | paraphrase | Может ли начальник пересадить меня за соседний станок в том же цехе без моего разрешения? | 72.1 | 276, 341.2, 309.2, 341.1-3, 404 |
| d025 | factual | Какая продолжительность отсутствия работника на рабочем месте без уважительных причин счит | 81 | 121, 128, 110, 108, 100 |
| d042 | paraphrase | Можно ли получить отступные или компенсацию при увольнении за косяки и нарушения на работе | 181.1 | 236, 394, 127, 165, 291 |
| d062 | paraphrase | Если я буду работать из дома временно, на какой самый долгий срок можно оформить такую уда | 312.1 | 110, 115, 284, 312.8, 108 |

## Answers

LLM: `tkrag-qwen2.5-14b`; refusal threshold (reranker score): 0.0066

| Metric | Value | n |
|---|---|---|
| Correct refusals on `no_answer` | 100% | 16 |
| False refusals on answerable questions | 1.3% | 76 |
| `answer_facts` recall (answerable, refusals count as 0; prepositions ignored) | 73.9% | 76 |
| `answer_facts` recall, strict (also demands prepositions and conjunctions) | 71.9% | 76 |
| Answers with every number and article confirmed by citations | 92.0% | 75 |

Outcome statuses: ok=69, refused=17, unverified=6

## Refusal threshold

Refuse when the best reranker score is below the threshold. The threshold with the best (correct refusals - false refusals) is **0.0068**. It is a suggestion fitted on this same golden set, not an independent estimate.

| Threshold | Correct refusals | False refusals |
|---|---|---|
| 0.0 | 0% | 0% |
| 0.0001 | 6.2% | 0% |
| 0.0003 | 18.8% | 0% |
| 0.0006 | 25.0% | 0% |
| 0.001 | 31.2% | 0% |
| 0.0013 | 37.5% | 0% |
| 0.0015 | 50.0% | 0% |
| 0.0016 | 56.2% | 0% |
| 0.0018 | 62.5% | 0% |
| 0.0022 | 68.8% | 0% |
| 0.0034 | 75.0% | 0% |
| 0.0046 | 81.2% | 0% |
| 0.0048 | 87.5% | 0% |
| 0.0051 | 93.8% | 0% |
| 0.0068 | 100% | 0% |
| 0.0086 | 100% | 1.3% |
| 0.0221 | 100% | 2.6% |
| 0.043 | 100% | 3.9% |
| 0.0894 | 100% | 5.3% |
| 0.1354 | 100% | 6.6% |
| 0.1643 | 100% | 7.9% |
| 0.2096 | 100% | 9.2% |
| 0.2614 | 100% | 10.5% |
| 0.2961 | 100% | 11.8% |
| 0.3463 | 100% | 13.2% |
| 0.4106 | 100% | 14.5% |
| 0.45 | 100% | 15.8% |
| 0.4939 | 100% | 17.1% |
| 0.5422 | 100% | 18.4% |
| 0.6101 | 100% | 19.7% |
| 0.668 | 100% | 21.1% |
| 0.6906 | 100% | 22.4% |
| 0.7169 | 100% | 23.7% |
| 0.7544 | 100% | 25.0% |
| 0.8208 | 100% | 26.3% |
| 0.875 | 100% | 27.6% |
| 0.8865 | 100% | 28.9% |
| 0.9008 | 100% | 30.3% |
| 0.9211 | 100% | 31.6% |
| 0.9302 | 100% | 32.9% |
| 0.943 | 100% | 34.2% |
| 0.9586 | 100% | 35.5% |
| 0.9639 | 100% | 36.8% |
| 0.9689 | 100% | 38.2% |
| 0.9735 | 100% | 39.5% |
| 0.9757 | 100% | 40.8% |
| 0.9774 | 100% | 42.1% |
| 0.98 | 100% | 43.4% |
| 0.9837 | 100% | 44.7% |
| 0.9872 | 100% | 46.1% |
| 0.9893 | 100% | 47.4% |
| 0.9901 | 100% | 48.7% |
| 0.9907 | 100% | 50.0% |
| 0.9916 | 100% | 51.3% |
| 0.9921 | 100% | 52.6% |
| 0.9923 | 100% | 53.9% |
| 0.9925 | 100% | 55.3% |
| 0.9932 | 100% | 56.6% |
| 0.9949 | 100% | 57.9% |
| 0.9969 | 100% | 60.5% |
| 0.998 | 100% | 61.8% |
| 0.998 | 100% | 63.2% |
| 0.9985 | 100% | 64.5% |
| 0.999 | 100% | 65.8% |
| 0.9993 | 100% | 67.1% |
| 0.9996 | 100% | 71.1% |
| 0.9997 | 100% | 75.0% |
| 0.9998 | 100% | 80.3% |
| 0.9999 | 100% | 82.9% |
| 1.0 | 100% | 93.4% |
| 1.0 | 100% | 93.4% |
