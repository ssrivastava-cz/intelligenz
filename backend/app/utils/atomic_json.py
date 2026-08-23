"""Crash-safe JSON writing, shared by anything persisting JSON straight
to disk. A plain `path.write_text(...)` can leave a truncated,
unparseable file behind if the process dies mid-write; this instead
writes to a temporary file in the same directory, flushes it to disk,
then atomically renames it over the target — `os.replace` is atomic on
both POSIX and Windows, so a reader never observes a partially-written
file.
"""
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            json.dump(data, tmp_file, indent=2)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
