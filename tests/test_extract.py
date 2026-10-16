from __future__ import annotations

import pytest

from iwir.extract import choose_answer
from iwir.models import Page
from iwir.textutil import supported

PARIS = "What is the capital of France?"


def page(url: str, text: str, pairs: list[tuple[str, str]] | None = None) -> Page:
    return Page(url=url, title="", text=text, pairs=pairs or [])


@pytest.mark.parametrize(
    ("question", "text", "pairs", "expected"),
    [
        (
            PARIS,
            "Capital: Paris\nFrance is a country in Western Europe. Its capital is Paris and its largest city is Paris.",
            [("Capital", "Paris"), ("Official language", "French")],
            "Paris",
        ),
        (
            "What is the chemical symbol for gold?",
            "Gold is a chemical element with the symbol Au and atomic number 79.",
            [],
            "Au",
        ),
        (
            "Who wrote the novel Nineteen Eighty-Four?",
            "Nineteen Eighty-Four is a dystopian novel written by George Orwell.",
            [],
            "George Orwell",
        ),
        (
            "What is the speed of light in vacuum in metres per second?",
            "The speed of light in vacuum is exactly 299792458 metres per second.",
            [],
            "299792458",
        ),
        (
            "How many planets are in the Solar System?",
            "There is one star in the Solar System. The Solar System has eight planets.",
            [],
            "8",
        ),
        (
            "In what year was the Python programming language first released?",
            "The Python programming language was first released in 1991 by Guido van Rossum.",
            [],
            "1991",
        ),
        (
            "What is the currency of Japan?",
            "Currency: Japanese yen\nJapan is an island country. The currency of Japan is the Japanese yen.",
            [("Currency", "Japanese yen")],
            "Japanese yen",
        ),
        (
            "What is the boiling point of water in degrees Celsius at standard atmospheric pressure?",
            "The boiling point of water is 100 degrees Celsius (212 degrees Fahrenheit) at standard atmospheric pressure.",
            [],
            "100",
        ),
        (
            "What is the largest planet in the Solar System?",
            "Jupiter is the largest planet in the Solar System.",
            [],
            "Jupiter",
        ),
        (
            "What are the three primary colors of light?",
            "The three primary colors of light are red, green, and blue.",
            [],
            "red, green, blue",
        ),
    ],
)
def test_direct_answers(question: str, text: str, pairs: list[tuple[str, str]], expected: str) -> None:
    chosen = choose_answer(question, [page("https://example.test/p", text, pairs)])
    assert chosen is not None
    assert chosen.answer == expected
    assert supported(expected, text)


def test_number_questions_do_not_return_a_name() -> None:
    text = "Planets: Mercury, Venus, Earth\nThe Solar System has eight planets."
    chosen = choose_answer(
        "How many planets are in the Solar System?",
        [page("https://example.test/sol", text, [("Planets", "Mercury, Venus, Earth")])],
    )
    assert chosen is not None
    assert chosen.answer == "8"


def test_name_must_look_like_a_name() -> None:
    text = "The capital is typically a city in that country. Paris is the capital of France."
    chosen = choose_answer(PARIS, [page("https://example.test/paris", text)])
    assert chosen is not None
    assert chosen.answer == "Paris"


def test_reverse_capital_sentence() -> None:
    text = "Paris is the capital and most populous city of France."
    chosen = choose_answer(PARIS, [page("https://example.test/paris", text)])
    assert chosen is not None
    assert chosen.answer == "Paris"
    assert choose_answer(PARIS, [page("https://example.test/x", "It is the capital of France.")]) is None


def test_search_query_keeps_of() -> None:
    from iwir.search.wikipedia import wikipedia_search_query
    from iwir.textutil import search_query

    assert search_query("What is the capital of France?") == "capital of France"
    assert search_query("In what year was the Python programming language first released?") == (
        "Python programming language first released"
    )
    assert wikipedia_search_query("capital of France") == "France"
    assert wikipedia_search_query("chemical symbol for gold") == "gold"
    assert wikipedia_search_query("boiling point of water in degrees Celsius") == "water"
    assert wikipedia_search_query("speed of light in vacuum") == "speed of light in vacuum"
    assert wikipedia_search_query("wrote the novel Nineteen Eighty-Four") == "Nineteen Eighty-Four"


def test_measurement_prefers_the_long_number() -> None:
    text = (
        "The refractive index of air is 1.0003. "
        "The speed of light in vacuum is exactly 299792458 metres per second."
    )
    chosen = choose_answer(
        "What is the speed of light in vacuum in metres per second?",
        [page("https://example.test/c", text)],
    )
    assert chosen is not None
    assert chosen.answer == "299792458"


def test_abstain_when_page_does_not_contain_the_value() -> None:
    text = "Welcome to the supplier portal. Sign in to continue."
    question = "What is the lead time in days for part ZX-9000-NO-SUCH on the supplier portal?"
    assert choose_answer(question, [page("https://example.test/portal", text)]) is None


def test_does_not_echo_the_question_entity() -> None:
    text = "France is a country in Western Europe."
    assert choose_answer(PARIS, [page("https://example.test/france", text)]) is None


def test_number_word_is_supported_as_a_digit() -> None:
    assert supported("8", "The Solar System has eight planets.")
    assert supported("eight", "There are 8 planets.")
