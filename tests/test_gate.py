from typing import Any

from tkrag.eval.answer_eval import calibrate_threshold, check_gate, fact_found, facts_recall
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
