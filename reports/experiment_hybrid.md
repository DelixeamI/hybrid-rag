# Hybrid experiment (first stage only, no reranker)

Dataset `full`, 76 answerable questions, split 39 dev / 37 test (stratified by type, seed 13). Configurations were compared on DEV; the winner and the baselines are then reported on TEST. Noise is large: one question moves Recall@1 by about 2.7 points.

## DEV: single channels

| Configuration | Recall@1 | Recall@5 | Recall@20 | MRR |
|---|---|---|---|---|
| dense | 71.8% | 79.5% | 94.9% | 0.766 |
| fts_cd | 20.5% | 43.6% | 66.7% | 0.300 |
| fts_cd_len | 23.1% | 48.7% | 76.9% | 0.358 |
| fts_cd_len_unique | 25.6% | 41.0% | 51.3% | 0.324 |
| fts_rank | 41.0% | 69.2% | 82.1% | 0.527 |
| title_only | 33.3% | 46.2% | 61.5% | 0.392 |
| bm25 | 48.7% | 82.1% | 87.2% | 0.623 |

## DEV: dense + one sparse channel fused with weighted RRF

| Configuration | Recall@1 | Recall@5 | Recall@20 | MRR |
|---|---|---|---|---|
| bm25, k=10, w_dense=3 | 69.2% | 84.6% | 92.3% | 0.772 |
| bm25, k=10, w_dense=2 | 66.7% | 89.7% | 94.9% | 0.768 |
| bm25, k=60, w_dense=3 | 66.7% | 89.7% | 94.9% | 0.761 |
| bm25, k=60, w_dense=2 | 66.7% | 87.2% | 94.9% | 0.756 |
| bm25, k=10, w_dense=1 | 64.1% | 89.7% | 94.9% | 0.750 |
| bm25, k=60, w_dense=1 | 64.1% | 87.2% | 97.4% | 0.737 |
| fts_cd_len, k=10, w_dense=3 | 61.5% | 84.6% | 92.3% | 0.726 |
| fts_cd_len_unique, k=10, w_dense=3 | 59.0% | 84.6% | 92.3% | 0.712 |
| fts_rank, k=10, w_dense=3 | 59.0% | 87.2% | 92.3% | 0.711 |
| fts_rank, k=10, w_dense=2 | 59.0% | 84.6% | 94.9% | 0.704 |
| fts_cd_len, k=10, w_dense=2 | 56.4% | 82.1% | 94.9% | 0.698 |
| fts_rank, k=60, w_dense=3 | 59.0% | 82.1% | 92.3% | 0.695 |
| title_only, k=10, w_dense=3 | 56.4% | 82.1% | 94.9% | 0.693 |
| fts_cd, k=10, w_dense=3 | 53.8% | 82.1% | 92.3% | 0.680 |
| fts_rank, k=60, w_dense=2 | 59.0% | 76.9% | 92.3% | 0.680 |

(top 15 of 36 by MRR; winner by the rule `best MRR, then Recall@5, then closest to the shipped setting`: **bm25, k=10, w_dense=3**)

## TEST

| Configuration | Recall@1 | Recall@5 | Recall@20 | MRR |
|---|---|---|---|---|
| dense alone | 78.4% | 89.2% | 94.6% | 0.832 |
| BM25 alone (true BM25 on Postgres lexemes) | 51.4% | 78.4% | 86.5% | 0.628 |
| today's hybrid (fts_cd, k=60, w_dense=1) | 37.8% | 67.6% | 94.6% | 0.521 |
| dev winner (bm25, k=10, w_dense=3) | 83.8% | 91.9% | 94.6% | 0.875 |
