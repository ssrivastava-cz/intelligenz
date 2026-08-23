"""Unit tests for `write_json_atomic` — the crash-safe write primitive
`FileSystemGenerationRepository` uses for every JSON file it writes.
"""
import json

import pytest

from app.utils.atomic_json import write_json_atomic


def test_write_json_atomic_writes_readable_json(tmp_path):
    path = tmp_path / "record.json"

    write_json_atomic(path, {"generation_id": "gen_001", "value": 1})

    assert json.loads(path.read_text(encoding="utf-8")) == {"generation_id": "gen_001", "value": 1}


def test_write_json_atomic_creates_parent_directories(tmp_path):
    path = tmp_path / "gen_001" / "user_question.json"

    write_json_atomic(path, {"ok": True})

    assert path.is_file()


def test_write_json_atomic_overwrites_an_existing_file(tmp_path):
    path = tmp_path / "record.json"
    write_json_atomic(path, {"value": "old"})

    write_json_atomic(path, {"value": "new"})

    assert json.loads(path.read_text(encoding="utf-8")) == {"value": "new"}


def test_write_json_atomic_leaves_no_temp_file_behind_on_success(tmp_path):
    path = tmp_path / "record.json"

    write_json_atomic(path, {"value": 1})

    assert list(tmp_path.iterdir()) == [path]


def test_write_json_atomic_leaves_the_original_file_untouched_if_serialization_fails(tmp_path):
    path = tmp_path / "record.json"
    write_json_atomic(path, {"value": "original"})

    class Unserializable:
        pass

    with pytest.raises(TypeError):
        write_json_atomic(path, {"value": Unserializable()})

    # The original file is exactly as it was — never truncated or
    # partially overwritten by the failed write.
    assert json.loads(path.read_text(encoding="utf-8")) == {"value": "original"}


def test_write_json_atomic_does_not_leave_a_temp_file_behind_on_failure(tmp_path):
    path = tmp_path / "record.json"

    class Unserializable:
        pass

    with pytest.raises(TypeError):
        write_json_atomic(path, {"value": Unserializable()})

    assert list(tmp_path.iterdir()) == []
