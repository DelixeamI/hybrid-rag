from tkrag.ingest.parser import Article, live_articles, parse_articles


def by_number(articles: list[Article]) -> dict[str, Article]:
    return {a.number: a for a in articles}


def test_hierarchy_is_attached_to_articles(articles: list[Article]) -> None:
    a115 = by_number(articles)["115"]
    assert a115.chapter == "Глава 19. Отпуска"
    assert a115.section == "Раздел V. ВРЕМЯ ОТДЫХА"
    assert a115.title == "Продолжительность ежегодного основного оплачиваемого отпуска"


def test_breadcrumbs_follow_the_spec_format(articles: list[Article]) -> None:
    assert by_number(articles)["115"].breadcrumbs == (
        "ТК РФ > Глава 19. Отпуска > Статья 115. "
        "Продолжительность ежегодного основного оплачиваемого отпуска"
    )


def test_chapter_changes_between_articles(articles: list[Article]) -> None:
    found = by_number(articles)
    assert (
        found["255"].chapter
        == "Глава 41. Особенности регулирования труда женщин, лиц с семейными обязанностями"
    )
    assert found["81"].chapter == "Глава 13. Расторжение трудового договора"
    assert found["178"].chapter is not None
    assert found["178"].chapter.startswith("Глава 27.")


def test_numbered_parts_are_split(articles: list[Article]) -> None:
    parts = by_number(articles)["81"].parts
    assert [p.number for p in parts] == ["1", "2", "3", "4", "5", "6"]
    assert parts[5].text.startswith("Не допускается увольнение")


def test_article_without_numbered_parts_has_one_unnumbered_part(articles: list[Article]) -> None:
    parts = by_number(articles)["115"].parts
    assert len(parts) == 1
    assert parts[0].number is None
    assert parts[0].text == (
        "Ежегодный основной оплачиваемый отпуск предоставляется работникам "
        "продолжительностью 28 календарных дней."
    )


def test_list_items_with_parenthesis_are_not_parts(articles: list[Article]) -> None:
    parts = by_number(articles)["77"].parts
    assert len(parts) == 1
    assert "1) соглашение сторон" in parts[0].text


def test_dotted_article_numbers(articles: list[Article]) -> None:
    assert "84.1" in by_number(articles)


def test_editorial_notes_are_removed(articles: list[Article]) -> None:
    text = " ".join(p.text for p in by_number(articles)["115"].parts)
    assert "в ред." not in text
    assert "N 90-ФЗ" not in text


def test_repealed_article_is_detected_and_not_indexed(articles: list[Article]) -> None:
    found = by_number(articles)
    assert found["83.1"].repealed
    assert "83.1" not in by_number(live_articles(articles))
    assert not found["83"].repealed


def test_synthetic_header_before_first_article_is_ignored(articles: list[Article]) -> None:
    assert all("SYNTHETIC" not in p.text for a in articles for p in a.parts)


def test_table_of_contents_duplicate_keeps_the_article_with_text() -> None:
    text = (
        "Глава 1. Общие положения\n"
        "Статья 1. Основные начала\n"
        "Статья 2. Принципы\n"
        "Глава 1. Общие положения\n"
        "Статья 1. Основные начала\n"
        "Работодатель обязан соблюдать закон.\n"
    )
    found = by_number(parse_articles(text))
    assert found["1"].parts[0].text == "Работодатель обязан соблюдать закон."
    assert found["2"].parts == ()


def test_part_numbers_must_continue_the_sequence() -> None:
    text = (
        "Статья 5. Тест\n"
        "1. Первая часть.\n"
        "Размер составляет не менее\n"
        "30. рублей в месяц.\n"
        "2. Вторая часть.\n"
    )
    parts = by_number(parse_articles(text))["5"].parts
    assert [p.number for p in parts] == ["1", "2"]
    assert "30. рублей" in parts[0].text


def test_hyphenated_article_numbers_of_the_real_code() -> None:
    text = (
        "Статья 341.1-1. Организации, имеющие право\nТекст статьи.\nСтатья 341.2. Другая\nТекст.\n"
    )
    assert set(by_number(parse_articles(text))) == {"341.1-1", "341.2"}


def test_unnumbered_paragraphs_become_implicit_parts_and_lists_stay_together() -> None:
    text = (
        "Статья 9. Тест\n"
        "Первый абзац статьи.\n"
        "Второй абзац содержит перечень:\n"
        "1) первый пункт;\n"
        "2) второй пункт.\n"
        "Третий абзац статьи.\n"
    )
    parts = by_number(parse_articles(text))["9"].parts
    assert [(p.number, p.implicit) for p in parts] == [("1", True), ("2", True), ("3", True)]
    assert "1) первый пункт;\n2) второй пункт." in parts[1].text


def test_single_paragraph_article_has_no_part_number() -> None:
    (part,) = by_number(parse_articles("Статья 9. Тест\nЕдинственный абзац.\n"))["9"].parts
    assert part.number is None
