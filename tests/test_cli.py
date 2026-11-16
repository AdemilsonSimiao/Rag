"""Tests for the command-line interface in src/cli.py."""

import json
from pathlib import Path

import pytest

from src.cli import Cli


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A tiny corpus, already indexed, with tmp_path as the cwd."""
    monkeypatch.chdir(tmp_path)
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    (raw / "lora.md").write_text("# LoRA\n\nLoad an adapter at runtime.\n")
    (raw / "cache.md").write_text("# Cache\n\nThe KV cache stores keys.\n")
    Cli().index()
    return tmp_path


def write_dataset(path: Path, text: str) -> None:
    """Write a dataset file with the given content."""
    path.write_text(text)


def test_search_dataset_writes_a_valid_file(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Results are saved under save_directory with the dataset's name."""
    dataset = project / "questions.json"
    write_dataset(dataset, json.dumps({"rag_questions": [
        {"question_id": "q1", "question": "How to load an adapter?"},
    ]}))
    Cli().search_dataset(str(dataset), 2, str(project / "out"))
    saved = json.loads((project / "out" / "questions.json").read_text())
    assert saved["k"] == 2
    first = saved["search_results"][0]
    assert first["question_id"] == "q1"
    assert first["retrieved_sources"][0]["file_path"] == "data/raw/lora.md"
    assert set(first["retrieved_sources"][0]) == {
        "file_path", "first_character_index", "last_character_index"
    }


def test_search_prints_json(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A single search prints the sources as JSON."""
    capsys.readouterr()
    Cli().search("kv cache", 1)
    printed = json.loads(capsys.readouterr().out)
    assert printed["retrieved_sources"][0]["file_path"] == "data/raw/cache.md"


@pytest.mark.parametrize("query, k", [("", 5), ("   ", 5), ("cache", 0)])
def test_search_bad_input_exits_cleanly(
    project: Path, query: str, k: int
) -> None:
    """Empty query or k <= 0 exits with status 1, no traceback."""
    with pytest.raises(SystemExit) as error:
        Cli().search(query, k)
    assert error.value.code == 1


def test_search_dataset_malformed_json_exits_cleanly(project: Path) -> None:
    """A broken dataset file is reported, not raised."""
    dataset = project / "bad.json"
    write_dataset(dataset, "{not json")
    with pytest.raises(SystemExit) as error:
        Cli().search_dataset(str(dataset), 5, str(project / "out"))
    assert error.value.code == 1


def test_search_dataset_missing_file_exits_cleanly(project: Path) -> None:
    """A missing dataset file is reported, not raised."""
    with pytest.raises(SystemExit):
        Cli().search_dataset("nope.json", 5, str(project / "out"))


def test_search_without_index_exits_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Searching before indexing tells the user to run index first."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        Cli().search("cache", 5)


def test_index_without_corpus_exits_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Indexing an empty folder is reported, not raised."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        Cli().index()
