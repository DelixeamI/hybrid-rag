"""Russian text normalisation shared by the validator and the answer evaluation.

Everything here is deterministic and dictionary-free on purpose: it must be easy to explain.
Known limits: no real lemmatisation (a crude prefix stem is used), and spelled-out numbers
are understood only up to 999.
"""

import re

_TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)?|[а-яa-z]+|%")
_NON_ALNUM_RE = re.compile(r"[^0-9a-zа-я]+")

_ONES = {
    "один": 1, "одна": 1, "одно": 1, "одного": 1, "одной": 1, "одному": 1,
    "два": 2, "две": 2, "двух": 2, "три": 3, "трех": 3, "четыре": 4, "четырех": 4,
    "пять": 5, "пяти": 5, "шесть": 6, "шести": 6, "семь": 7, "семи": 7,
    "восемь": 8, "восьми": 8, "девять": 9, "девяти": 9,
}  # fmt: skip
_TEENS = {
    "десять": 10, "десяти": 10, "одиннадцать": 11, "двенадцать": 12, "тринадцать": 13,
    "четырнадцать": 14, "пятнадцать": 15, "шестнадцать": 16, "семнадцать": 17,
    "восемнадцать": 18, "девятнадцать": 19,
}  # fmt: skip
_TENS = {
    "двадцать": 20, "двадцати": 20, "тридцать": 30, "тридцати": 30, "сорок": 40, "сорока": 40,
    "пятьдесят": 50, "пятидесяти": 50, "шестьдесят": 60, "шестидесяти": 60,
    "семьдесят": 70, "семидесяти": 70, "восемьдесят": 80, "восьмидесяти": 80,
    "девяносто": 90, "девяноста": 90,
}  # fmt: skip
_HUNDREDS = {
    "сто": 100, "ста": 100, "двести": 200, "двухсот": 200, "триста": 300, "трехсот": 300,
    "четыреста": 400, "пятьсот": 500, "шестьсот": 600, "семьсот": 700, "восемьсот": 800,
    "девятьсот": 900,
}  # fmt: skip

_UNIT_FORMS: dict[str, frozenset[str]] = {
    "day": frozenset({"день", "дня", "дню", "днем", "дне", "дни", "дней", "дням", "днях"}),
    "hour": frozenset({"час", "часа", "часу", "часом", "часе", "часы", "часов", "часам"}),
    "week": frozenset({"неделя", "недели", "неделю", "неделе", "неделей", "недель", "неделям"}),
    "month": frozenset({"месяц", "месяца", "месяцу", "месяцем", "месяце", "месяцы", "месяцев"}),
    "year": frozenset({"год", "года", "году", "годом", "годе", "годы", "лет", "годам", "годов"}),
    "percent": frozenset({"%", "процент", "процента", "процентов", "проценты"}),
    "ruble": frozenset({"рубль", "рубля", "рублей", "рублю", "рубли", "руб"}),
}
_UNIT_BY_FORM = {form: unit for unit, forms in _UNIT_FORMS.items() for form in forms}
_ADJECTIVE_PREFIXES = ("календарн", "рабоч", "полн", "полугод")
_ARTICLE_REF_RE = re.compile(
    r"\bст(?:атья|атьи|атье|атью|атьей|атей|атьям|атьях|\.)\s*(\d+(?:[.-]\d+)*)"
)

Quantity = tuple[str, str]  # (canonical value, unit class), e.g. ("28", "day")


def normalize(text: str) -> str:
    """Lower-case, fold ё to е, collapse whitespace."""
    return " ".join(text.lower().replace("ё", "е").split())


def squash(text: str) -> str:
    """Normalise and drop everything except letters and digits (for quote matching)."""
    return _NON_ALNUM_RE.sub("", normalize(text))


def _number_value(token: str) -> int | None:
    for table in (_HUNDREDS, _TENS, _TEENS, _ONES):
        if token in table:
            return table[token]
    return None


def _parse_number_words(tokens: list[str], start: int) -> tuple[int, int] | None:
    """Parse e.g. ['двадцать', 'восемь'] -> (28, 2). Returns (value, tokens consumed)."""
    total, i = 0, start
    for table in (_HUNDREDS, _TENS):
        if i < len(tokens) and tokens[i] in table:
            total += table[tokens[i]]
            i += 1
    if i < len(tokens) and (tokens[i] in _TEENS or tokens[i] in _ONES):
        total += _TEENS.get(tokens[i]) or _ONES[tokens[i]]
        i += 1
    return (total, i - start) if i > start else None


def canonical_tokens(text: str) -> list[str]:
    """Tokenise; digits use a dot decimal separator, spelled-out numbers become digits."""
    raw = _TOKEN_RE.findall(normalize(text))
    out: list[str] = []
    i = 0
    while i < len(raw):
        parsed = _parse_number_words(raw, i)
        if parsed is not None:
            out.append(str(parsed[0]))
            i += parsed[1]
            continue
        token = raw[i]
        out.append(token.replace(",", ".") if token[0].isdigit() else token)
        i += 1
    return out


def _format_value(token: str) -> str:
    return f"{float(token):g}"


def extract_quantities(text: str) -> set[Quantity]:
    """Every number that is directly followed by a known unit, e.g. '28 календарных дней'."""
    tokens = canonical_tokens(text)
    found: set[Quantity] = set()
    for i, token in enumerate(tokens):
        unit = _UNIT_BY_FORM.get(token)
        if unit is None:
            continue
        j = i - 1
        while j >= 0 and tokens[j].startswith(_ADJECTIVE_PREFIXES):
            j -= 1
        if j >= 0 and tokens[j][0].isdigit():
            found.add((_format_value(tokens[j]), unit))
    return found


def extract_article_refs(text: str) -> set[str]:
    """Article numbers mentioned in the text: 'статья 115', 'ст. 81', 'статьей 114'."""
    return set(_ARTICLE_REF_RE.findall(normalize(text)))


# Prepositions, conjunctions and particles that carry no fact on their own. "не" is NOT here.
FUNCTION_WORDS = frozenset(
    [
        "в",
        "во",
        "на",
        "по",
        "до",
        "за",
        "из",
        "от",
        "с",
        "со",
        "к",
        "ко",
        "о",
        "об",
        "у",
        "и",
        "а",
        "но",
        "или",
        "при",
        "для",
        "над",
        "под",
        "про",
        "через",
        "без",
        "между",
        "же",
        "ли",
        "бы",
        "течение",  # only occurs in the preposition "в течение"
    ]
)


def stem(token: str) -> str:
    """Crude prefix stem: enough to match 'отпуск'/'отпуска'/'отпуске', not a lemmatiser.

    Units collapse to one name per class ('день'/'дней' -> 'day', 'год'/'лет' -> 'year'), which
    a prefix cannot do for short irregular forms.
    """
    if token[0].isdigit():
        return _format_value(token)
    if token in _UNIT_BY_FORM:
        return _UNIT_BY_FORM[token]
    return token[:5] if len(token) > 5 else token


def stems(text: str) -> set[str]:
    return {stem(t) for t in canonical_tokens(text)}


def content_stems(text: str) -> set[str]:
    """Like `stems` but without prepositions and conjunctions."""
    found = {stem(t) for t in canonical_tokens(text) if t not in FUNCTION_WORDS}
    return found or stems(text)
