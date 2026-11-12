# Evaluation results

Dataset: **full** — 534 articles, 710 chunks, 92 questions (`golden.jsonl`).

Embeddings `intfloat/multilingual-e5-small`, reranker `BAAI/bge-reranker-v2-m3`, RRF k=10 (dense weight 3), rerank top-20 -> top-5. Recall@k: a question counts as found when ANY expected article is among the top-k distinct articles (alternatives); for `multi_article` it is the SHARE of the required articles found, and MRR uses the rank where the last required article appears.

## Retrieval

| Mode | Recall@1 | Recall@5 | Recall@10 | MRR | latency p50, ms | latency p95, ms |
|---|---|---|---|---|---|---|
| Postgres FTS (ts_rank_cd) | 19.7% | 39.5% | 46.1% | 0.251 | 13.7 | 185.3 |
| BM25 (Postgres lexemes) | 46.1% | 77.6% | 82.9% | 0.551 | 4.8 | 7.5 |
| dense (e5) | 69.1% | 80.3% | 88.8% | 0.696 | 22.8 | 36.0 |
| hybrid (dense + BM25, RRF) | 71.1% | 84.9% | 88.8% | 0.727 | 26.9 | 36.5 |
| hybrid + rerank | 80.3% | 90.8% | 92.8% | 0.816 | 2990.7 | 4254.8 |

**Recall@1 by question type**

| Type | n | Postgres FTS (ts_rank_cd) | BM25 (Postgres lexemes) | dense (e5) | hybrid (dense + BM25, RRF) | hybrid + rerank |
|---|---|---|---|---|---|---|
| factual | 33 | 33.3% | 72.7% | 93.9% | 97.0% | 97.0% |
| multi_article | 10 | 10.0% | 30.0% | 45.0% | 40.0% | 40.0% |
| paraphrase | 33 | 9.1% | 24.2% | 51.5% | 54.5% | 75.8% |

**Recall@5 by question type**

| Type | n | Postgres FTS (ts_rank_cd) | BM25 (Postgres lexemes) | dense (e5) | hybrid (dense + BM25, RRF) | hybrid + rerank |
|---|---|---|---|---|---|---|
| factual | 33 | 60.6% | 97.0% | 97.0% | 97.0% | 97.0% |
| multi_article | 10 | 20.0% | 70.0% | 70.0% | 75.0% | 90.0% |
| paraphrase | 33 | 24.2% | 60.6% | 66.7% | 75.8% | 84.8% |

**MRR by question type**

| Type | n | Postgres FTS (ts_rank_cd) | BM25 (Postgres lexemes) | dense (e5) | hybrid (dense + BM25, RRF) | hybrid + rerank |
|---|---|---|---|---|---|---|
| factual | 33 | 0.432 | 0.827 | 0.955 | 0.970 | 0.970 |
| multi_article | 10 | 0.000 | 0.184 | 0.186 | 0.217 | 0.349 |
| paraphrase | 33 | 0.147 | 0.385 | 0.593 | 0.640 | 0.803 |

### Misses

Questions not fully found in the top 5 of `hybrid_rerank` (nothing found, or for `multi_article` at least one required article missing): 8 of 76.

| id | type | question | expected | found (top 5) |
|---|---|---|---|---|
| d006 | paraphrase | Могут ли городские власти или чиновники министерств представлять руководство бюджетных учр | 34 | 6, 407, 33, 412, 281.1 |
| d020 | paraphrase | Может ли начальник пересадить меня за соседний станок в том же цехе без моего разрешения? | 72.1 | 276, 341.2, 309.2, 341.1-3, 404 |
| d025 | factual | Какая продолжительность отсутствия работника на рабочем месте без уважительных причин счит | 81 | 121, 110, 193, 108, 100 |
| d036 | paraphrase | Может ли фирма сама решить давать сотрудникам дополнительные дни отдыха, если на это есть  | 116 | 305, 111, 119, 186, 153 |
| d042 | paraphrase | Можно ли получить отступные или компенсацию при увольнении за косяки и нарушения на работе | 181.1 | 236, 394, 127, 165, 291 |
| d062 | paraphrase | Если я буду работать из дома временно, на какой самый долгий срок можно оформить такую уда | 312.1 | 110, 115, 284, 312.8, 108 |
| d067 | multi_article | Сохраняется ли за работником должность на время ежегодного отпуска и какова минимальная пр | 114, 115 | 263, 115, 128, 124, 117 |
| d075 | multi_article | Какой документ оформляется при недостижении согласия на коллективных переговорах и на како | 38, 39 | 39, 40, 403, 374, 405 |

## Answers

LLM: `tkrag-qwen2.5-14b`; refusal threshold (reranker score): 0.0069

| Metric | All | DEV (threshold fitted here) | TEST (held out) |
|---|---|---|---|
| Correct refusals on `no_answer` | 16/16 (100%) | 8/8 (100%) | 8/8 (100%) |
| False refusals on answerable questions | 0/76 (0%) | 0/39 (0%) | 0/37 (0%) |
| `answer_facts` recall (refusals count as 0; prepositions ignored) | 73.2% | 69.2% | 77.5% |
| `answer_facts` recall, strict (also demands prepositions and conjunctions) | 71.3% | 65.4% | 77.5% |
| Grounded answers: numbers, articles and quotes are in the cited text (of the answers given, not of all questions) | 71/76 (93.4%) | 38/39 (97.4%) | 33/37 (89.2%) |

Outcome statuses (all): ok=71, refused=16, unverified=5

**How to read these numbers.**

- *Grounded answers* is a syntactic check: the numbers with units, the article numbers and the quotes of the answer occur in the cited fragments. It is not the share of legally correct answers; a grounded answer can still draw a wrong conclusion from the right article. It also counts only the answers that were given.
- *`answer_facts` recall* is lexical overlap with the expected facts (words and numbers after a crude stemming). It misses a correct answer in other words and can accept a wrong answer that contains the same words.
- *Correct refusals*: the threshold was fitted on the DEV half only, so the TEST column is a held-out estimate. TEST holds few `no_answer` questions, so it is a rough one.
- The retrieval parameters (RRF k and dense weight) were chosen on the DEV half of the answerable questions (`reports/experiment_hybrid.md`); the TEST half was not used.

## Refusal threshold

Refuse when the best reranker score is below the threshold. The threshold with the best (correct refusals - false refusals) on the DEV half (47 questions) is **0.0069**. The TEST half was not used to choose it.

| Threshold | Correct refusals | False refusals |
|---|---|---|
| 0.0 | 0% | 0% |
| 0.0006 | 12.5% | 0% |
| 0.001 | 25.0% | 0% |
| 0.0013 | 37.5% | 0% |
| 0.0016 | 50.0% | 0% |
| 0.0022 | 62.5% | 0% |
| 0.0036 | 75.0% | 0% |
| 0.0048 | 87.5% | 0% |
| 0.0069 | 100% | 0% |
| 0.0685 | 100% | 2.6% |
| 0.1568 | 100% | 5.1% |
| 0.2096 | 100% | 7.7% |
| 0.2683 | 100% | 10.3% |
| 0.3673 | 100% | 12.8% |
| 0.4756 | 100% | 15.4% |
| 0.61 | 100% | 17.9% |
| 0.7833 | 100% | 20.5% |
| 0.875 | 100% | 23.1% |
| 0.8865 | 100% | 25.6% |
| 0.9008 | 100% | 28.2% |
| 0.9215 | 100% | 30.8% |
| 0.9462 | 100% | 33.3% |
| 0.9639 | 100% | 35.9% |
| 0.9712 | 100% | 38.5% |
| 0.9774 | 100% | 41.0% |
| 0.9822 | 100% | 43.6% |
| 0.9872 | 100% | 46.2% |
| 0.9899 | 100% | 48.7% |
| 0.9916 | 100% | 51.3% |
| 0.9921 | 100% | 53.8% |
| 0.9924 | 100% | 56.4% |
| 0.9943 | 100% | 59.0% |
| 0.997 | 100% | 61.5% |
| 0.998 | 100% | 64.1% |
| 0.9988 | 100% | 66.7% |
| 0.9996 | 100% | 74.4% |
| 0.9997 | 100% | 76.9% |
| 0.9998 | 100% | 82.1% |
| 0.9999 | 100% | 87.2% |
| 1.0 | 100% | 92.3% |
| 1.0 | 100% | 92.3% |
