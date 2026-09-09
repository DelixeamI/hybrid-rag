import pytest

from tkrag.eval.metrics import mean, percentile, ranked_articles, recall_at_k, reciprocal_rank
from tkrag.retrieval.rrf import fuse, rrf_scores
from tkrag.retrieval.search import build_tsquery
from tkrag.retrieval.types import Hit


def hit(chunk_id: int, article: str = "1", score: float = 0.0) -> Hit:
    return Hit(chunk_id, f"{article}#0", article, "t", "b", "text", [], score)


def test_rrf_scores_match_the_formula() -> None:
    scores = rrf_scores([[1, 2, 3], [3, 1]], k=60)
    assert scores[1] == pytest.approx(1 / 61 + 1 / 62)
    assert scores[2] == pytest.approx(1 / 62)
    assert scores[3] == pytest.approx(1 / 63 + 1 / 61)


def test_rrf_rewards_agreement_between_rankings() -> None:
    # 9 is only second in both lists, but beats 1 and 2 that are first in just one list.
    fused = fuse([[hit(1), hit(9), hit(3)], [hit(2), hit(9), hit(4)]])
    assert fused[0].chunk_id == 9


def test_small_k_favours_top_ranks_large_k_favours_agreement() -> None:
    # 10 is first in one list only; 20 is second in both lists.
    top = [[10, 20], [30, 20]]
    small_k = rrf_scores(top, k=1)
    large_k = rrf_scores(top, k=1000)
    assert small_k[10] / small_k[20] > large_k[10] / large_k[20]


def test_rrf_is_deterministic_on_ties() -> None:
    fused = fuse([[hit(2)], [hit(1)]])
    assert [h.chunk_id for h in fused] == [1, 2]


def test_recall_and_mrr_on_known_ranks() -> None:
    ranked = ["7", "115", "80"]
    assert recall_at_k(ranked, {"115"}, 1) == 0.0
    assert recall_at_k(ranked, {"115"}, 5) == 1.0
    assert recall_at_k(ranked, {"1", "80"}, 5) == 1.0  # any expected article counts
    assert reciprocal_rank(ranked, {"115"}) == 0.5
    assert reciprocal_rank(ranked, {"nope"}) == 0.0


def test_ranked_articles_dedupes_chunks_of_one_article() -> None:
    hits = [hit(1, "81"), hit(2, "81"), hit(3, "80")]
    assert ranked_articles(hits) == ["81", "80"]


def test_percentile_nearest_rank() -> None:
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert percentile(values, 50) == 30.0
    assert percentile(values, 95) == 50.0
    assert percentile([], 50) == 0.0
    assert mean([]) == 0.0


def test_tsquery_is_or_of_unique_words_and_safe() -> None:
    assert build_tsquery("Сколько дней отпуска? отпуска") == "сколько | дней | отпуска"
    assert build_tsquery("a'; DROP TABLE chunks; --") == "drop | table | chunks"
    assert build_tsquery("?!") == ""


def test_multi_article_needs_every_required_article() -> None:
    # The question needs articles 115 and 125; the search found only 115.
    ranked = ["115", "7", "80"]
    expected = {"115", "125"}
    assert recall_at_k(ranked, expected, 5) == 1.0  # alternatives: any one counts (old behaviour)
    assert recall_at_k(ranked, expected, 5, need_all=True) == 0.5  # required: half an answer
    assert recall_at_k(["115", "125"], expected, 1, need_all=True) == 0.5
    assert recall_at_k(["125", "7", "115"], expected, 5, need_all=True) == 1.0
    assert recall_at_k(ranked, set(), 5, need_all=True) == 0.0


def test_multi_article_reciprocal_rank_is_the_rank_of_the_last_required_article() -> None:
    expected = {"115", "125"}
    assert reciprocal_rank(["115", "125"], expected, need_all=True) == 0.5
    assert reciprocal_rank(["115", "7", "125"], expected, need_all=True) == pytest.approx(1 / 3)
    assert reciprocal_rank(["115", "7", "80"], expected, need_all=True) == 0.0  # 125 missing
    assert reciprocal_rank(["115", "7", "125"], expected) == 1.0  # alternatives: first hit
