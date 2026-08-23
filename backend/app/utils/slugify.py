import re

from app.core.exceptions import ValidationError

_UNSAFE_CHARS = re.compile(r"[^a-zA-Z0-9._-]+")


def safe_path_segment(value: str) -> str:
    """Validates and normalizes a string for safe use as a single
    filesystem path segment.

    Used for `feature` / `generation_id` values that become folder
    names. Rejects (rather than silently mangling) anything containing
    a path separator or a `.`/`..` traversal segment, so a crafted
    value like `../../etc` can't escape the storage root.
    """
    stripped = value.strip()
    if not stripped or "/" in stripped or "\\" in stripped or stripped in {".", ".."}:
        raise ValidationError(f"'{value}' is not a valid identifier.")

    slug = _UNSAFE_CHARS.sub("-", stripped).strip("-")
    if not slug:
        raise ValidationError(f"'{value}' is not a valid identifier.")
    return slug
