# Hybrid experiment (first stage only, no reranker)

Dataset `full`, 76 answerable questions, split 39 dev / 37 test (stratified by type, seed 13). Configurations were compared on DEV; the winner and the baselines are then reported on TEST. Noise is large: one question moves Recall@1 by about 2.7 points.

## DEV: single channels

| Configuration | Recall@1 | Recall@5 | Recall@20 | MRR |
|---|---|---|---|---|
| dense | 65.4% | 74.4% | 93.6% | 0.657 |
| fts_cd | 19.2% | 41.0% | 62.8% | 0.268 |
| fts_cd_len | 20.5% | 46.2% | 75.6% | 0.307 |
| fts_cd_len_unique | 20.5% | 37.2% | 50.0% | 0.235 |
| fts_rank | 37.2% | 66.7% | 80.8% | 0.461 |
| title_only | 28.2% | 42.3% | 61.5% | 0.306 |
| bm25 | 43.6% | 79.5% | 87.2% | 0.543 |

## DEV: dense + one sparse channel fused with weighted RRF

| Configuration | Recall@1 | Recall@5 | Recall@20 | MRR |
|---|---|---|---|---|
| bm25, k=10, w_dense=3 | 64.1% | 79.5% | 91.0% | 0.678 |
| bm25, k=10, w_dense=2 | 61.5% | 84.6% | 93.6% | 0.675 |
| bm25, k=60, w_dense=3 | 61.5% | 84.6% | 93.6% | 0.668 |
| bm25, k=60, w_dense=2 | 61.5% | 83.3% | 93.6% | 0.666 |
| bm25, k=10, w_dense=1 | 59.0% | 87.2% | 93.6% | 0.663 |
| bm25, k=60, w_dense=1 | 59.0% | 83.3% | 96.2% | 0.648 |
| fts_rank, k=10, w_dense=2 | 53.8% | 80.8% | 93.6% | 0.618 |
| fts_rank, k=10, w_dense=3 | 53.8% | 82.1% | 91.0% | 0.618 |
| fts_cd_len, k=10, w_dense=3 | 55.1% | 79.5% | 91.0% | 0.618 |
| fts_rank, k=60, w_dense=3 | 53.8% | 78.2% | 91.0% | 0.604 |
| fts_cd_len_unique, k=10, w_dense=3 | 52.6% | 79.5% | 92.3% | 0.604 |
| fts_rank, k=60, w_dense=2 | 53.8% | 73.1% | 91.0% | 0.594 |
| fts_cd_len, k=10, w_dense=2 | 50.0% | 76.9% | 93.6% | 0.590 |
| fts_rank, k=10, w_dense=1 | 51.3% | 79.5% | 93.6% | 0.589 |
| title_only, k=10, w_dense=3 | 50.0% | 76.9% | 93.6% | 0.587 |

(top 15 of 36 by MRR; winner by the rule `best MRR, then Recall@5, then closest to the shipped setting`: **bm25, k=10, w_dense=3**)

## TEST

| Configuration | Recall@1 | Recall@5 | Recall@20 | MRR |
|---|---|---|---|---|
| dense alone | 73.0% | 86.5% | 94.6% | 0.754 |
| BM25 alone (true BM25 on Postgres lexemes) | 48.6% | 75.7% | 85.1% | 0.568 |
| today's hybrid (fts_cd, k=60, w_dense=1) | 36.5% | 64.9% | 90.5% | 0.479 |
| dev winner (bm25, k=10, w_dense=3) | 78.4% | 90.5% | 94.6% | 0.795 |
