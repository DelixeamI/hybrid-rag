# Evaluation results

Dataset: **full** — 534 articles, 710 chunks, 92 questions (`golden.jsonl`).

Embeddings `intfloat/multilingual-e5-small`, reranker `BAAI/bge-reranker-v2-m3`, RRF k=10 (dense weight 3), rerank top-20 -> top-5. Recall@k: a question counts as found when ANY expected article is among the top-k distinct articles (alternatives); for `multi_article` it is the SHARE of the required articles found, and MRR uses the rank where the last required article appears.

## Retrieval

| Mode | Recall@1 | Recall@5 | Recall@10 | MRR | latency p50, ms | latency p95, ms |
|---|---|---|---|---|---|---|
| Postgres FTS (ts_rank_cd) | 19.7% | 39.5% | 46.1% | 0.251 | 10.8 | 139.3 |
| BM25 (Postgres lexemes) | 46.1% | 77.6% | 82.9% | 0.551 | 3.7 | 5.5 |
| dense (e5) | 69.1% | 80.3% | 88.8% | 0.696 | 23.4 | 31.2 |
| hybrid (dense + BM25, RRF) | 71.1% | 84.9% | 88.8% | 0.727 | 27.7 | 33.5 |
| hybrid + rerank | 80.3% | 90.8% | 92.8% | 0.816 | 397.6 | 476.2 |

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

LLM: `tkrag-qwen2.5-14b`; refusal threshold (reranker score): 0.0066

| Metric | Value | n |
|---|---|---|
| Correct refusals on `no_answer` | 100% | 16 |
| False refusals on answerable questions | 0% | 76 |
| `answer_facts` recall (answerable, refusals count as 0; prepositions ignored) | 73.2% | 76 |
| `answer_facts` recall, strict (also demands prepositions and conjunctions) | 71.3% | 76 |
| Grounded answers: numbers, articles and quotes are in the cited text (share of the answers given, not of all questions) | 93.4% | 76 |

Outcome statuses: ok=71, refused=16, unverified=5

**How to read these numbers.**

- *Grounded answers* is a syntactic check: the numbers with units, the article numbers and the quotes of the answer occur in the cited fragments. It is not the share of legally correct answers; a grounded answer can still draw a wrong conclusion from the right article. It also counts only the answers that were given.
- *`answer_facts` recall* is lexical overlap with the expected facts (words and numbers after a crude stemming). It misses a correct answer in other words and can accept a wrong answer that contains the same words.
- *Correct refusals* depend on a threshold that was chosen on this same golden set, so they are optimistic for new questions. Only a set that was not used to choose any parameter or threshold gives an honest estimate (see `eval --frozen` in the README).

## Refusal threshold

Refuse when the best reranker score is below the threshold. The threshold with the best (correct refusals - false refusals) is **0.0066**. It is a suggestion fitted on this same golden set, not an independent estimate.

| Threshold | Correct refusals | False refusals |
|---|---|---|
| 0.0 | 0% | 0% |
| 0.0001 | 6.2% | 0% |
| 0.0003 | 18.8% | 0% |
| 0.0006 | 25.0% | 0% |
| 0.001 | 31.2% | 0% |
| 0.0012 | 37.5% | 0% |
| 0.0014 | 43.8% | 0% |
| 0.0015 | 56.2% | 0% |
| 0.0016 | 62.5% | 0% |
| 0.0017 | 68.8% | 0% |
| 0.0022 | 75.0% | 0% |
| 0.0034 | 81.2% | 0% |
| 0.0046 | 87.5% | 0% |
| 0.0048 | 93.8% | 0% |
| 0.0066 | 100% | 0% |
| 0.0086 | 100% | 1.3% |
| 0.0221 | 100% | 2.6% |
| 0.043 | 100% | 3.9% |
| 0.0894 | 100% | 5.3% |
| 0.1354 | 100% | 6.6% |
| 0.1431 | 100% | 7.9% |
| 0.1644 | 100% | 9.2% |
| 0.2096 | 100% | 10.5% |
| 0.2614 | 100% | 11.8% |
| 0.2961 | 100% | 13.2% |
| 0.3463 | 100% | 14.5% |
| 0.4106 | 100% | 15.8% |
| 0.45 | 100% | 17.1% |
| 0.4939 | 100% | 18.4% |
| 0.5422 | 100% | 19.7% |
| 0.6101 | 100% | 21.1% |
| 0.668 | 100% | 22.4% |
| 0.6906 | 100% | 23.7% |
| 0.7169 | 100% | 25.0% |
| 0.7544 | 100% | 26.3% |
| 0.8208 | 100% | 27.6% |
| 0.875 | 100% | 28.9% |
| 0.8865 | 100% | 30.3% |
| 0.9008 | 100% | 31.6% |
| 0.9215 | 100% | 32.9% |
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
