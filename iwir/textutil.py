"""Text helpers shared by extraction, verification, and scoring."""

from __future__ import annotations

import re

STOPWORDS = {
    "a",
    "an",
    "the",
    "of",
    "in",
    "on",
    "for",
    "to",
    "and",
    "or",
    "what",
    "which",
    "who",
    "whom",
    "whose",
    "when",
    "where",
    "why",
    "how",
    "many",
    "much",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "do",
    "does",
    "did",
    "with",
    "from",
    "by",
    "at",
    "as",
    "that",
    "this",
    "these",
    "those",
    "it",
    "its",
    "into",
    "about",
    "please",
    "tell",
    "me",
    "there",
    "their",
    "than",
    "then",
    "per",
    "was",
    "were",
}

NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
}

_WORD = re.compile(r"[a-z0-9]+", re.I)


def norm_text(value: str) -> str:
    """Casefold and strip punctuation so 'Paris,' and 'paris' compare equal.

    Decimal points between digits are kept (``3.14``). A period at the end of
    a sentence is not, so ``blue.`` still contains ``blue``.
    """
    text = value.casefold().replace("\u00a0", " ")
    text = text.replace("’", "'").replace("`", "'").replace(",", "")
    text = re.sub(r"(?<!\d)\.(?!\d)", " ", text)
    text = re.sub(r"[^\w\s.+-]", " ", text, flags=re.UNICODE)
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip(" .")


def tokens(value: str) -> list[str]:
    return [tok.casefold() for tok in _WORD.findall(value)]


_CANON = {
    "wrote": "author",
    "written": "author",
    "writing": "author",
    "write": "author",
    "authored": "author",
    "author": "author",
    "released": "release",
    "release": "release",
    "appeared": "release",
    "appear": "release",
    "introduced": "release",
    "introduction": "release",
    "published": "publish",
    "publication": "publish",
    "planets": "planet",
    "planet": "planet",
    "colours": "color",
    "colors": "color",
    "colour": "color",
    "color": "color",
    "metres": "meter",
    "meters": "meter",
    "metre": "meter",
    "meter": "meter",
    "languages": "language",
    "language": "language",
    "cities": "city",
    "city": "city",
    "currencies": "currency",
    "currency": "currency",
}


def content_terms(value: str) -> list[str]:
    seen: list[str] = []
    for tok in tokens(value):
        if tok in STOPWORDS or len(tok) <= 1:
            continue
        if tok not in seen:
            seen.append(tok)
    return seen


def canon_terms(value: str) -> set[str]:
    return {_CANON.get(term, term) for term in content_terms(value)}


def head_term(value: str) -> str | None:
    """First content word, lightly canonicalized (planets -> planet)."""
    terms = content_terms(value)
    if not terms:
        return None
    return _CANON.get(terms[0], terms[0])


def intent_of(question: str) -> str:
    q = question.casefold()
    if re.search(r"\b(what are|which are|name the|name all|list)\b", q):
        return "list"
    if re.search(
        r"\b(how many|how much|population|price|number of|atomic number|"
        r"speed of|boiling point|distance|percentage|how old)\b",
        q,
    ):
        return "number"
    if re.search(
        r"\b(when|what year|which year|what date|founded|released|born|published)\b",
        q,
    ):
        return "date"
    if re.search(
        r"\b(who|whom|author|written by|invented by|capital|currency|symbol|"
        r"name of|largest|smallest|highest|language|painted)\b",
        q,
    ):
        return "name"
    return "fact"


def answer_forms(answer: str) -> list[str]:
    """Surface form plus number-word / digit twins ('eight' and '8')."""
    normalized = norm_text(answer)
    forms = [normalized] if normalized else []
    if normalized in NUMBER_WORDS:
        forms.append(NUMBER_WORDS[normalized])
    inverse = {digit: word for word, digit in NUMBER_WORDS.items()}
    if normalized in inverse:
        forms.append(inverse[normalized])
    unique: list[str] = []
    for form in forms:
        if form and form not in unique:
            unique.append(form)
    return unique


def _tokens_in_order(needles: list[str], hay_tokens: list[str]) -> bool:
    if not needles:
        return False
    index = 0
    for tok in hay_tokens:
        if index < len(needles) and tok == needles[index]:
            index += 1
    return index == len(needles)


def supported(answer: str, text: str) -> bool:
    """True when the answer (or its number-word twin) actually occurs in text."""
    if not answer or not answer.strip() or not text:
        return False
    hay_tokens = norm_text(text).split()
    hay = " ".join(hay_tokens)
    for form in answer_forms(answer):
        needles = [tok for tok in form.split() if tok not in {"and", "or"}]
        if not needles:
            continue
        if " ".join(needles) in hay or _tokens_in_order(needles, hay_tokens):
            return True
    return False


def echoes_question(question: str, answer: str) -> bool:
    """Reject spans that only repeat the question ('France' for capital of France)."""
    raw = tokens(answer)
    if not raw:
        return True
    if all(tok.isdigit() for tok in raw):
        return False
    significant = [tok for tok in raw if tok not in STOPWORDS and not tok.isdigit()]
    # "It" / "the" survived a pattern but are not an answer. Digits already returned.
    if not significant:
        return True
    return set(significant) <= set(content_terms(question))


def same_answer(left: str, right: str) -> bool:
    if norm_text(left) == norm_text(right):
        return True
    return bool(set(answer_forms(left)) & set(answer_forms(right)))


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"[\n\r]+|(?<=[.!?])\s+(?=[A-Z0-9])", text)
    sentences: list[str] = []
    for part in parts:
        cleaned = re.sub(r"\s+", " ", part).strip()
        if len(cleaned) >= 2:
            sentences.append(cleaned)
    return sentences


_LEADING_QUESTION = re.compile(
    r"^(?:please\s+)?(?:(?:in|on|at)\s+)?"
    r"(?:what|who|when|where|which|how many|how much|how)\s+"
    r"(?:(?:year|date|time|day)\s+)?"
    r"(?:is|was|are|were|did|does|do)?\s*",
    re.I,
)


def search_query(question: str) -> str:
    """Drop the question words and keep the rest, including 'of'.

    Wikipedia's search API treats ``What is the capital of France?`` as a bag
    of words and drifts onto 'capital punishment'. ``capital of France`` does not.
    """
    text = question.strip().rstrip("?").strip()
    stripped = _LEADING_QUESTION.sub("", text)
    stripped = re.sub(r"^(?:the|a|an)\s+", "", stripped, flags=re.I)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    return stripped or text


def keyword_query(question: str) -> str:
    terms = content_terms(question)
    return " ".join(terms)
