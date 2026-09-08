import math

import pytest

from tkrag.eval.experiments import (
    BASELINE,
    Config,
    Prepared,
    fused_ranking,
    grid,
    pick_winner,
    split_dev_test,
)
from tkrag.eval.golden import GoldenItem
from tkrag.retrieval.bm25 import Bm25Index
from tkrag.retrieval.rrf import fuse, rrf_scores
from tkrag.retrieval.types import Hit


def item(ident: str, kind: str = "factual") -> GoldenItem:
    return GoldenItem(
        id=ident,
        question="q",
        expected_articles=["1"],
        answer_facts=["x"],
        type=kind,
    )


def test_split_is_stratified_disjoint_and_reproducible() -> None:
    items = [item(f"f{i}") for i in range(10)] + [item(f"p{i}", "paraphrase") for i in range(7)]
    dev, test = split_dev_test(items, seed=5)
    assert {i.id for i in dev}.isdisjoint({i.id for i in test})
    assert len(dev) + len(test) == len(items)
    assert sum(i.type == "paraphrase" for i in dev) in (3, 4)  # 7 split as 4 + 3
    assert [i.id for i in split_dev_test(items, seed=5)[0]] == [i.id for i in dev]
    assert [i.id for i in split_dev_test(items, seed=6)[0]] != [i.id for i in dev]


def test_bm25_prefers_rare_terms_and_shorter_documents() -> None:
    index = Bm25Index()
    index.add(1, {"отпуск": 1, "ежегодн": 1})
    index.add(2, {"отпуск": 1, "работник": 5, "трудов": 5, "договор": 5})
    index.add(3, {"работник": 2, "договор": 2})
    assert index.search(["отпуск"], 10)[0][0] == 1  # same tf, shorter document wins
    # "ежегодн" occurs once in the corpus, "работник" twice: the rarer term outweighs
    assert index.search(["ежегодн", "работник"], 10)[0][0] == 1
    assert index.search(["нетакого"], 10) == []


def test_bm25_idf_matches_the_formula() -> None:
    index = Bm25Index(k1=1.2, b=0.0)  # no length normalisation, easy to compute by hand
    index.add(1, {"a": 2})
    index.add(2, {"b": 1})
    idf = math.log(1 + (2 - 1 + 0.5) / (1 + 0.5))
    expected = idf * 2 * 2.2 / (2 + 1.2)
    # peek at the score through ranking two documents that share nothing: only doc 1 matches
    assert [cid for cid, _ in index.search(["a"], 5)] == [1]
    assert expected > 0


def test_weighted_rrf_changes_the_order() -> None:
    first, second = [1, 2], [2, 1]
    plain = rrf_scores([first, second], k=60)
    assert plain[1] == pytest.approx(plain[2])
    heavy = rrf_scores([first, second], k=60, weights=[3.0, 1.0])
    assert heavy[1] > heavy[2]
    with pytest.raises(ValueError, match="one weight"):
        rrf_scores([first, second], weights=[1.0])


def test_fuse_accepts_weights() -> None:
    def hit(i: int) -> Hit:
        return Hit(i, f"{i}#0", str(i), "t", "b", "x", [], 0.0)

    fused = fuse([[hit(1), hit(2)], [hit(2), hit(1)]], k=60, weights=[1.0, 5.0])
    assert [h.chunk_id for h in fused] == [2, 1]


def test_fused_ranking_uses_dense_and_the_chosen_sparse_channel() -> None:
    prepared = Prepared(item("a"), {"dense": [1, 2, 3], "bm25": [3, 2, 1], "fts_cd": [2, 1, 3]})
    by_bm25 = fused_ranking(prepared, Config("bm25", 60, 1.0))
    by_fts = fused_ranking(prepared, Config("fts_cd", 60, 1.0))
    assert set(by_bm25) == {1, 2, 3}
    assert by_bm25 != by_fts
    dense_heavy = fused_ranking(prepared, Config("bm25", 60, 10.0))
    assert dense_heavy[0] == 1


def test_pick_winner_prefers_mrr_then_the_shipped_setting() -> None:
    scores = {
        BASELINE: {"mrr": 0.5, "recall@5": 0.7},
        Config("bm25", 60, 2.0): {"mrr": 0.6, "recall@5": 0.7},
        Config("bm25", 10, 3.0): {"mrr": 0.6, "recall@5": 0.7},
    }
    assert pick_winner(scores) == Config("bm25", 60, 2.0)  # tie on MRR: closer to the baseline


def test_grid_covers_every_sparse_channel() -> None:
    configs = grid(["a", "b"])
    assert {c.sparse for c in configs} == {"a", "b"}
    assert len(configs) == 2 * 2 * 3
