from tkrag.generation.validator import CitedSource, locate_quote, validate_answer
from tkrag.retrieval.types import Hit

TEXT_115 = "Ежегодный основной оплачиваемый отпуск предоставляется работникам продолжительностью 28 календарных дней."
TEXT_80 = (
    "1. Работник имеет право расторгнуть трудовой договор, предупредив об этом работодателя "
    "в письменной форме не позднее чем за две недели. См. статью 84.1 настоящего Кодекса."
)


def hit(article: str, title: str, text: str) -> Hit:
    crumbs = f"ТК РФ > Глава 1. Тест > Статья {article}. {title}"
    return Hit(1, f"{article}#0", article, title, crumbs, text, [], 1.0)


H115 = hit("115", "Продолжительность отпуска", TEXT_115)
H80 = hit("80", "Расторжение по инициативе работника", TEXT_80)


def src(n: int, h: Hit, quote: str | None = None) -> CitedSource:
    return CitedSource(n, h, quote or h.text)


def test_a_faithful_answer_passes() -> None:
    verdict = validate_answer(
        "Отпуск длится 28 календарных дней [1].", [src(1, H115, "28 календарных дней")]
    )
    assert verdict.ok, verdict.issues


def test_number_not_in_the_chunk_is_rejected() -> None:
    verdict = validate_answer("Отпуск длится 30 календарных дней [1].", [src(1, H115)])
    assert not verdict.ok
    assert any("30 day" in issue for issue in verdict.issues)


def test_same_number_with_another_unit_is_rejected() -> None:
    verdict = validate_answer("Отпуск длится 28 рабочих часов [1].", [src(1, H115)])
    assert not verdict.ok


def test_spelled_out_number_is_checked_too() -> None:
    ok = validate_answer(
        "Предупредить нужно за две недели [1].", [src(1, H80, "не позднее чем за две недели")]
    )
    bad = validate_answer("Предупредить нужно за три недели [1].", [src(1, H80, "за две недели")])
    assert ok.ok, ok.issues
    assert not bad.ok


def test_number_from_an_uncited_chunk_is_rejected() -> None:
    # 28 days is in H115, but the answer cites only H80.
    verdict = validate_answer("Отпуск длится 28 календарных дней [2].", [src(2, H80, "две недели")])
    assert not verdict.ok


def test_article_number_must_be_cited_or_referenced() -> None:
    cited = [src(1, H80, "не позднее чем за две недели")]
    assert validate_answer("Это статья 80 [1].", cited).ok
    assert validate_answer("Порядок оформления в статье 84.1 [1].", cited).ok  # referenced in chunk
    bad = validate_answer("Это описано в статье 81 [1].", cited)
    assert not bad.ok
    assert any("article 81" in issue for issue in bad.issues)


def test_quote_must_be_in_the_chunk() -> None:
    verdict = validate_answer(
        "Отпуск 28 календарных дней [1].", [src(1, H115, "отпуск длится 40 дней")]
    )
    assert not verdict.ok
    assert any("quote does not match the fragment" in issue for issue in verdict.issues)


def test_quote_match_ignores_case_and_punctuation() -> None:
    verdict = validate_answer(
        "Отпуск 28 календарных дней [1].", [src(1, H115, "ПРОДОЛЖИТЕЛЬНОСТЬЮ 28 календарных дней")]
    )
    assert verdict.ok, verdict.issues


def test_no_citations_or_no_markers_is_rejected() -> None:
    assert not validate_answer("Отпуск 28 календарных дней [1].", []).ok
    verdict = validate_answer("Отпуск 28 календарных дней.", [src(1, H115, "28 календарных дней")])
    assert any("no [n] references" in issue for issue in verdict.issues)


def test_marker_to_uncited_fragment_is_rejected() -> None:
    verdict = validate_answer(
        "Отпуск 28 календарных дней [3].", [src(1, H115, "28 календарных дней")]
    )
    assert any("uncited" in issue for issue in verdict.issues)


def test_marker_digits_are_not_read_as_quantities() -> None:
    assert validate_answer("Положен отпуск [1] дней.", [src(1, H115, "28 календарных дней")]).ok


def test_a_reworded_quote_is_replaced_by_the_real_passage() -> None:
    verdict = validate_answer(
        "Отпуск 28 календарных дней [1].",
        [src(1, H115, "Ежегодный основной отпуск предоставляется на 28 календарных дней")],
    )
    assert verdict.ok, verdict.issues
    assert verdict.quotes == [TEXT_115]  # what the user sees is verbatim chunk text


def test_a_verbatim_quote_is_kept_as_the_model_wrote_it() -> None:
    verdict = validate_answer(
        "Отпуск 28 календарных дней [1].", [src(1, H115, "продолжительностью 28 календарных дней")]
    )
    assert verdict.quotes == ["продолжительностью 28 календарных дней"]


def test_an_ellipsis_may_skip_words_but_not_invent_them() -> None:
    quote = "Ежегодный основной оплачиваемый отпуск ... 28 календарных дней"
    assert locate_quote(quote, TEXT_115) == quote
    assert locate_quote("Ежегодный отпуск ... 35 календарных дней", TEXT_115) is None


def test_an_unrelated_quote_is_not_repaired() -> None:
    assert locate_quote("Работодатель обязан выплатить премию каждому работнику", TEXT_115) is None
    verdict = validate_answer(
        "Отпуск 28 календарных дней [1].",
        [src(1, H115, "Работодатель обязан выплатить премию каждому работнику")],
    )
    assert not verdict.ok


def test_repair_picks_the_best_line_of_a_multi_line_chunk() -> None:
    text = (
        "Первая строка про другое.\n"
        "Работодатель обязан предупредить работника за два месяца.\n"
        "Третья."
    )
    found = locate_quote("работодатель должен предупредить работника за два месяца", text)
    assert found == "Работодатель обязан предупредить работника за два месяца."


def test_weeks_and_days_are_the_same_duration() -> None:
    weeks = validate_answer("Нужно за 14 дней [1].", [src(1, H80, "не позднее чем за две недели")])
    assert weeks.ok, weeks.issues
    days = hit("77", "Т", "Срок составляет 14 дней.")
    reverse = validate_answer("Нужно за две недели [1].", [src(1, days, "Срок составляет 14 дней")])
    assert reverse.ok, reverse.issues
    assert not validate_answer("Нужно за 15 дней [1].", [src(1, H80, "за две недели")]).ok
