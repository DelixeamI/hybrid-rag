from tkrag.textnorm import (
    canonical_tokens,
    extract_article_refs,
    extract_quantities,
    squash,
    stems,
)


def test_digits_with_unit() -> None:
    assert extract_quantities("Отпуск 28 календарных дней") == {("28", "day")}


def test_spelled_out_numbers_become_digits() -> None:
    assert canonical_tokens("двадцать восемь дней")[:1] == ["28"]
    assert extract_quantities("не позднее чем за две недели") == {("2", "week")}
    assert extract_quantities("сто восемьдесят дней") == {("180", "day")}


def test_units_and_word_forms() -> None:
    text = "5 часов, 3 месяца, 2 года, 10 процентов, 15 рублей, 7%"
    assert extract_quantities(text) == {
        ("5", "hour"),
        ("3", "month"),
        ("2", "year"),
        ("10", "percent"),
        ("15", "ruble"),
        ("7", "percent"),
    }


def test_decimal_numbers() -> None:
    assert extract_quantities("2,5 месяца") == {("2.5", "month")}


def test_number_without_unit_is_ignored() -> None:
    assert extract_quantities("в статье 115 части 2") == set()


def test_article_refs() -> None:
    text = "Согласно статье 115 и ст. 84.1, а также статьей 140 настоящего Кодекса"
    assert extract_article_refs(text) == {"115", "84.1", "140"}


def test_squash_ignores_case_yo_and_punctuation() -> None:
    assert squash("Отпуск,  ПРЕДОСТАВЛЯЕТСЯ — ежегодно!") == squash(
        "отпуск предоставляется ежегодно"
    )
    assert squash("всё") == squash("все")


def test_stems_match_word_forms() -> None:
    assert stems("отпуска") == stems("отпуске") == stems("отпуск")
    assert stems("28 календарных дней") >= {"28", "day"}  # unit forms collapse


def test_article_refs_with_hyphenated_numbers() -> None:
    assert extract_article_refs("в соответствии со статьей 341.1-1 настоящего Кодекса.") == {
        "341.1-1"
    }
    assert extract_article_refs("см. статью 81.") == {"81"}
