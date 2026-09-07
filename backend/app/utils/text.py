from pathlib import Path
from typing import Any, Iterable

_MAX_HEADING_LENGTH = 80
_MAX_ACRONYM_LENGTH = 5


def title_from_filename(filename: str) -> str:
    """Humanizes a filename into a display title — the fallback title
    for formats with no embedded title of their own.
    """
    stem = Path(filename).stem
    cleaned = stem.replace("_", " ").replace("-", " ").strip()
    return cleaned.title() if cleaned else filename


def looks_like_heading(line: str) -> bool:
    """A simple, format-agnostic "is this line a heading?" heuristic.

    Shared by parsers for formats with no native heading markup to
    rely on (unlike Markdown's `#` or Word's paragraph styles) — TXT
    and, within a page's extracted text, PDF. A line qualifies if it's
    short, doesn't trail off mid-sentence, and is either ALL CAPS
    ("ROLE PERMISSIONS"), ends with a colon ("Validation Rules:"), or
    is Title Case with every word capitalized ("Contact Creation").

    The Title Case rule requires at least two words, to avoid treating
    an ordinary capitalized sentence-opener ("Thanks for waiting") as
    a heading — those get excluded by the punctuation check anyway
    when they end a sentence, but a lone capitalized word with no
    trailing punctuation is too weak a signal on its own.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > _MAX_HEADING_LENGTH:
        return False
    if stripped.endswith((".", ",", ";")):
        return False
    if stripped.endswith(":"):
        return True
    if stripped.isupper():
        return True
    return _is_title_case(stripped)


def _is_title_case(line: str) -> bool:
    words = [word for word in line.split() if word[0].isalpha()]
    if len(words) < 2:
        return False
    return all(_looks_like_title_word(word) for word in words)


def _looks_like_title_word(word: str) -> bool:
    """True for a word that plausibly belongs in a natural-language
    heading — ordinary Title Case ("Patient", "Creation") or a short,
    purely-alphabetic ALL-CAPS acronym ("ID", "API"). False for an
    alphanumeric code or identifier ("P10001", "PLAN-A") that merely
    happens to start with an uppercase letter.

    The previous version of this check only looked at a word's first
    character (`word[0].isupper()`), so a data value like "P10001" or
    "PLAN-A" — uppercase first letter, arbitrary digits/punctuation
    after it — passed as if it were a real Title Case word, causing
    lines like "Patient ID P10001" or "Plan ID PLAN-A" to be
    misclassified as section headings (confirmed against the real
    Source of Truth PDFs, which contain exactly this pattern).
    """
    if any(character.isdigit() for character in word):
        return False
    if word.isalpha() and word.isupper():
        return len(word) <= _MAX_ACRONYM_LENGTH
    return word[0].isupper() and word[1:].islower()


def is_blank_value(value: Any) -> bool:
    """Whether a single cell/field value carries no meaningful content.

    `None` and an empty/whitespace-only string are blank. `0`, `0.0`,
    and `False` are never blank — they're meaningful values that merely
    happen to be falsy, not the absence of one. Anything else is
    stringified and checked for whitespace-only content, so a value
    that arrives as a non-string (e.g. already-parsed numeric/boolean
    types from a source other than `csv.DictReader`, which always
    yields `str | None`) is still handled safely.
    """
    if value is None:
        return True
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return False
    return str(value).strip() == ""


def is_blank_row(values: Iterable[Any]) -> bool:
    """Whether every value in a row carries no meaningful content — see
    `is_blank_value`. An empty row (no values at all) counts as blank.
    """
    return all(is_blank_value(value) for value in values)
