from typing import Any

import pytest

from tkrag.eval.answer_eval import calibrate_threshold, check_gate, fact_found, facts_recall
from tkrag.eval.golden import GoldenItem
from tkrag.generation.schemas import Answer, AskResult


def results(recall5: float, correct_refusals: float) -> dict[str, Any]:
    return {
        "retrieval": {"hybrid_rerank": {"overall": {"recall@5": recall5}}},
        "answers": {"correct_refusal_rate": correct_refusals},
    }


THRESHOLDS = {"recall_at_5": 0.9, "correct_refusal_rate": 0.75}


def test_gate_passes_at_or_above_the_floors() -> None:
    assert check_gate(results(0.9, 0.75), THRESHOLDS)["ok"]


def test_gate_fails_when_recall_drops() -> None:
    gate = check_gate(results(0.8, 1.0), THRESHOLDS)
    assert not gate["ok"]
    assert [c["name"] for c in gate["checks"] if not c["ok"]] == ["recall_at_5"]


def test_gate_fails_when_refusals_drop() -> None:
    gate = check_gate(results(1.0, 0.5), THRESHOLDS)
    assert not gate["ok"]
    assert [c["name"] for c in gate["checks"] if not c["ok"]] == ["correct_refusal_rate"]


def test_fact_matching_ignores_word_forms_and_number_words() -> None:
    assert fact_found("28 календарных дней", "Отпуск длится двадцать восемь календарных дней [1].")
    assert fact_found("две недели", "Нужно предупредить за 2 недели.")
    assert not fact_found("28 календарных дней", "Отпуск длится 30 календарных дней.")


def test_fact_matching_is_lenient_about_prepositions_and_unit_forms() -> None:
    assert fact_found("до одного года", "Максимальный срок: один год [1].")
    assert not fact_found("до одного года", "Максимальный срок: один год [1].", strict=True)
    assert fact_found("в течение одного месяца", "Не более одного месяца.")
    assert fact_found("5 лет", "Срок составляет 5 года")  # unit forms collapse to one name
    assert fact_found("в день увольнения", "Выплата в день увольнения работника")
    assert not fact_found("в течение трех месяцев", "в течение двух месяцев")  # numbers still count


def test_a_negation_is_never_ignored() -> None:
    assert not fact_found("не допускается", "увольнение допускается")
    assert fact_found("не допускается", "увольнение не допускается")


def test_a_fact_of_only_function_words_falls_back_to_all_words() -> None:
    assert fact_found("до", "срок до месяца")
    assert not fact_found("до", "срок месяц")


def test_facts_recall_counts_only_accepted_answers() -> None:
    ok = AskResult(status="ok", answer=Answer(answer="Отпуск 28 календарных дней [1]."))
    unverified = AskResult(status="unverified", answer=Answer(answer="Отпуск 28 календарных дней"))
    refused = AskResult(status="refused", answer=Answer(answer="нет", refused=True))
    facts = ["28 календарных дней", "отпуск"]
    assert facts_recall(facts, ok) == 1.0
    assert facts_recall(["28 календарных дней", "ипотека"], ok) == 0.5
    assert facts_recall(facts, unverified) == 0.0
    assert facts_recall(facts, refused) == 0.0


def test_calibration_separates_clean_scores() -> None:
    scored = [(True, 0.001), (True, 0.002), (False, 0.5), (False, 0.9)]
    calibration = calibrate_threshold(scored)
    assert 0.002 < calibration.threshold <= 0.5
    best = max(calibration.curve, key=lambda c: c["youden"])
    assert best["correct_refusal"] == 1.0 and best["false_refusal"] == 0.0


def _golden(ident: str, kind: str) -> GoldenItem:
    return GoldenItem.model_validate(
        {
            "id": ident,
            "question": f"q {ident}",
            "expected_articles": [] if kind == "no_answer" else ["1"],
            "answer_facts": [] if kind == "no_answer" else ["x"],
            "type": kind,
        }
    )


def _result(status: str) -> AskResult:
    answer = Answer(answer="a", refused=status == "refused")
    return AskResult(status=status, answer=answer)


def test_summarize_keeps_counts_next_to_rates() -> None:
    from tkrag.eval.answer_eval import summarize

    results = [
        (_golden("n1", "no_answer"), _result("refused")),
        (_golden("n2", "no_answer"), _result("ok")),  # an invented answer to an off-topic question
        (_golden("a1", "factual"), _result("ok")),
        (_golden("a2", "factual"), _result("unverified")),
        (_golden("a3", "factual"), _result("refused")),  # a false refusal
    ]
    s = summarize(results)
    assert (s["n_correct_refusals"], s["n_no_answer"]) == (1, 2)
    assert (s["n_false_refusals"], s["n_answerable"]) == (1, 3)
    assert (s["n_grounded"], s["n_answered"]) == (2, 3)  # ok + ok + unverified were given
    assert s["verified_rate"] == pytest.approx(2 / 3)


def test_answerable_split_does_not_change_when_no_answer_questions_are_added() -> None:
    from tkrag.eval.experiments import split_dev_test

    answerable = (
        [_golden(f"f{i}", "factual") for i in range(9)]
        + [_golden(f"m{i}", "multi_article") for i in range(5)]
        + [_golden(f"p{i}", "paraphrase") for i in range(8)]
    )
    refusals = [_golden(f"n{i}", "no_answer") for i in range(6)]
    dev_a, test_a = split_dev_test(answerable)
    dev_all, test_all = split_dev_test(answerable + refusals)
    # the hybrid parameters were chosen on the DEV half of the answerable questions only
    assert {i.id for i in dev_all if i.type != "no_answer"} == {i.id for i in dev_a}
    assert {i.id for i in test_all if i.type != "no_answer"} == {i.id for i in test_a}
    assert sum(i.type == "no_answer" for i in dev_all) == 3  # no_answer is split in half too


def test_refusal_threshold_comes_from_the_dataset_unless_the_environment_sets_it() -> None:
    from tkrag.components import refusal_threshold
    from tkrag.config import Settings
    from tkrag.datasets import FIXTURE, FULL_REFUSAL_THRESHOLD

    assert refusal_threshold(Settings(_env_file=None), FIXTURE) == FIXTURE.refusal_threshold
    assert refusal_threshold(Settings(_env_file=None, refusal_threshold=0.42), FIXTURE) == 0.42
    assert FIXTURE.refusal_threshold != FULL_REFUSAL_THRESHOLD  # scores depend on the corpus


def test_the_report_shows_dev_and_test_columns_with_counts() -> None:
    from tkrag.eval.report import _answer_section

    side = {
        "n_no_answer": 8, "n_answerable": 37, "n_answered": 37, "n_correct_refusals": 8,
        "n_false_refusals": 0, "n_grounded": 35, "correct_refusal_rate": 1.0,
        "false_refusal_rate": 0.0, "facts_recall": 0.7, "facts_recall_strict": 0.68,
        "verified_rate": 35 / 37, "status_counts": {"ok": 35, "refused": 8},
    }  # fmt: skip
    answers = {
        "llm_model": "m",
        "refusal_threshold": 0.01,
        **side,
        "by_split": {"dev": side, "test": side},
    }
    text = "\n".join(_answer_section(answers))
    assert "TEST (held out)" in text and "DEV (threshold fitted here)" in text
    assert "8/8 (100%)" in text and "0/37 (0%)" in text and "35/37" in text
