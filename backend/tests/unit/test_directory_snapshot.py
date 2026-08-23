"""Unit tests proving the directory-snapshot utility correctly detects
additions, modifications, and deletions — the mechanism
`tests/conftest.py` relies on to guard real application data during the
test suite. If this utility were buggy (e.g. silently missed a content
change), the isolation guard could pass even when it shouldn't.
"""
from tests.directory_snapshot import diff_snapshots, snapshot_directory


def test_snapshot_of_nonexistent_directory_is_empty(tmp_path):
    assert snapshot_directory(tmp_path / "does-not-exist") == {}


def test_snapshot_captures_every_file_and_its_exact_content(tmp_path):
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "b.txt").write_text("world", encoding="utf-8")

    snapshot = snapshot_directory(tmp_path)

    assert snapshot == {"a.txt": b"hello", "nested/b.txt": b"world"}


def test_snapshot_ignores_subdirectories_as_entries_but_walks_into_them(tmp_path):
    (tmp_path / "empty_dir").mkdir()
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")

    snapshot = snapshot_directory(tmp_path)

    assert snapshot == {"a.txt": b"hello"}  # no entry for the empty directory itself


def test_snapshot_detects_an_added_file(tmp_path):
    before = snapshot_directory(tmp_path)

    (tmp_path / "new.txt").write_text("new", encoding="utf-8")
    after = snapshot_directory(tmp_path)

    assert before != after
    assert "added" in diff_snapshots(before, after)


def test_snapshot_detects_a_removed_file(tmp_path):
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    before = snapshot_directory(tmp_path)

    (tmp_path / "a.txt").unlink()
    after = snapshot_directory(tmp_path)

    assert before != after
    assert "removed" in diff_snapshots(before, after)


def test_snapshot_detects_modified_content_even_with_the_same_filename(tmp_path):
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    before = snapshot_directory(tmp_path)

    (tmp_path / "a.txt").write_text("goodbye", encoding="utf-8")
    after = snapshot_directory(tmp_path)

    assert before != after
    assert "modified" in diff_snapshots(before, after)


def test_snapshot_detects_a_newly_created_root_directory(tmp_path):
    root = tmp_path / "was-not-there"
    before = snapshot_directory(root)

    root.mkdir()
    (root / "file.txt").write_text("surprise", encoding="utf-8")
    after = snapshot_directory(root)

    assert before == {}
    assert before != after
    assert "added" in diff_snapshots(before, after)


def test_diff_snapshots_reports_no_differences_when_identical(tmp_path):
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    snapshot = snapshot_directory(tmp_path)

    assert diff_snapshots(snapshot, snapshot) == "no differences"
