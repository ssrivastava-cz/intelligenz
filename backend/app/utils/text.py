from pathlib import Path

_MAX_HEADING_LENGTH = 80


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
    return all(word[0].isupper() for word in words)
