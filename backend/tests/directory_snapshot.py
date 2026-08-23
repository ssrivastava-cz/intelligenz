"""A pure directory-content snapshot utility — no tests live here, just
a helper `tests/conftest.py` uses to prove the test suite never touches
real application data, and that `tests/unit/test_directory_snapshot.py`
exercises directly to prove the mechanism itself is trustworthy.
"""
from pathlib import Path


def snapshot_directory(root: Path) -> dict[str, bytes]:
    """Every file's path (relative to `root`) mapped to its exact bytes.

    Returns an empty dict if `root` doesn't exist — "nothing there" and
    "an empty directory" are treated the same, which is exactly what we
    want: if a test *creates* `root` where none existed before, the
    resulting snapshot differs from the empty baseline and the isolation
    guard fails, exactly as it should.
    """
    if not root.is_dir():
        return {}
    # `.as_posix()` keeps keys stable across platforms (Windows would
    # otherwise produce backslash-separated keys).
    return {
        path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }


def diff_snapshots(before: dict[str, bytes], after: dict[str, bytes]) -> str:
    """A human-readable description of what changed between two
    snapshots — added/removed/modified file paths — for a clear
    assertion failure message instead of an opaque dict-inequality.
    """
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    modified = sorted(path for path in set(before) & set(after) if before[path] != after[path])

    parts = []
    if added:
        parts.append(f"added: {added}")
    if removed:
        parts.append(f"removed: {removed}")
    if modified:
        parts.append(f"modified: {modified}")
    return "; ".join(parts) if parts else "no differences"
