"""Tests for the indexer in src/indexer.py."""

from pathlib import Path

import pytest

from src.indexer import build_index, list_files, load_index


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A small data/raw tree; the tests run with tmp_path as the cwd."""
    monkeypatch.chdir(tmp_path)
    raw = tmp_path / "data" / "raw" / "proj"
    (raw / "docs").mkdir(parents=True)
    (raw / "a.py").write_text("def load_adapter():\n    return 1\n")
    (raw / "docs" / "lora.md").write_text("# LoRA\n\nLoad an adapter.\n")
    (raw / "notes.txt").write_text("plain text notes\n")
    (raw / "image.png").write_bytes(b"\x89PNG")
    return tmp_path


def test_list_files_keeps_only_py_md_txt(corpus: Path) -> None:
    """Other file types are ignored; paths are relative with '/'."""
    assert list_files("data/raw") == [
        "data/raw/proj/a.py",
        "data/raw/proj/docs/lora.md",
        "data/raw/proj/notes.txt",
    ]


def test_build_and_load_index(corpus: Path) -> None:
    """The index is saved under index_dir and finds the right file."""
    count = build_index("data/raw", "data/processed", 2000)
    assert count == 3
    index = load_index("data/processed")
    top = index.search("load an adapter", 1)[0][0]
    assert top.file_path.startswith("data/raw/proj/")


def test_chunks_never_exceed_the_limit(corpus: Path) -> None:
    """With a small max size, every chunk respects it."""
    (corpus / "data" / "raw" / "proj" / "big.py").write_text("x = 1\n" * 500)
    build_index("data/raw", "data/processed", 300)
    index = load_index("data/processed")
    assert all(
        c.last_character_index - c.first_character_index <= 300
        for c in index.chunks
    )


def test_build_index_without_files_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty or missing raw folder raises ValueError."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        build_index("data/raw", "data/processed", 2000)


def test_build_index_invalid_size_raises(corpus: Path) -> None:
    """A non-positive max_chunk_size raises ValueError."""
    with pytest.raises(ValueError):
        build_index("data/raw", "data/processed", 0)


def test_load_index_missing_raises(tmp_path: Path) -> None:
    """Loading before indexing gives a FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_index(str(tmp_path / "nothing"))
