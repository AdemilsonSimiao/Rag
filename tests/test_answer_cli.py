"""Tests for the answer commands of the CLI, with a fake model."""

import json
from pathlib import Path
from typing import Dict, List

import pytest

from src.cli import Cli


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> List[str]:
    """Replace the real model by a fake one that records its prompts."""
    prompts: List[str] = []

    def fake_load(model_name: str) -> object:
        def generate(messages: List[Dict[str, str]]) -> str:
            prompts.append(messages[1]["content"])
            return "fake answer"
        return generate

    monkeypatch.setattr("src.generator.load_model", fake_load)
    return prompts


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A tiny indexed corpus, with tmp_path as the cwd."""
    monkeypatch.chdir(tmp_path)
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    (raw / "lora.md").write_text("# LoRA\n\nLoad an adapter at runtime.\n")
    Cli().index()
    return tmp_path


def write_results(path: Path, sources: List[Dict[str, object]]) -> None:
    """Write a StudentSearchResults file with one question."""
    path.write_text(json.dumps({"k": 5, "search_results": [{
        "question_id": "q1",
        "question": "How to load an adapter?",
        "retrieved_sources": sources,
    }]}))


def test_answer_dataset_writes_answers(
    project: Path, calls: List[str]
) -> None:
    """Each question gets an answer, with its sources and the same k."""
    results = project / "results.json"
    write_results(results, [{
        "file_path": "data/raw/lora.md",
        "first_character_index": 0,
        "last_character_index": 40,
    }])
    Cli().answer_dataset(str(results), str(project / "out"))
    saved = json.loads((project / "out" / "results.json").read_text())
    assert saved["k"] == 5
    item = saved["search_results"][0]
    assert item["answer"] == "fake answer"
    assert item["question_id"] == "q1"
    assert item["retrieved_sources"][0]["file_path"] == "data/raw/lora.md"
    assert "Load an adapter" in calls[0]


def test_answer_dataset_skips_unreadable_sources(
    project: Path, calls: List[str]
) -> None:
    """A source whose file is gone is skipped, and with none left the
    model is not called."""
    results = project / "results.json"
    write_results(results, [{
        "file_path": "data/raw/missing.md",
        "first_character_index": 0,
        "last_character_index": 10,
    }])
    Cli().answer_dataset(str(results), str(project / "out"))
    saved = json.loads((project / "out" / "results.json").read_text())
    assert "No relevant sources" in saved["search_results"][0]["answer"]
    assert calls == []


def test_answer_dataset_malformed_json_exits_cleanly(
    project: Path, calls: List[str]
) -> None:
    """A broken results file is reported, not raised."""
    bad = project / "bad.json"
    bad.write_text("{oops")
    with pytest.raises(SystemExit) as error:
        Cli().answer_dataset(str(bad), str(project / "out"))
    assert error.value.code == 1


def test_answer_dataset_missing_file_exits_cleanly(
    project: Path, calls: List[str]
) -> None:
    """A missing results file is reported, not raised."""
    with pytest.raises(SystemExit):
        Cli().answer_dataset("nope.json", str(project / "out"))


def test_model_that_cannot_load_exits_cleanly(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If torch/transformers or the model are missing, no traceback."""
    def broken(model_name: str) -> object:
        raise ImportError("No module named 'torch'")

    monkeypatch.setattr("src.generator.load_model", broken)
    results = project / "results.json"
    write_results(results, [{
        "file_path": "data/raw/lora.md",
        "first_character_index": 0,
        "last_character_index": 40,
    }])
    with pytest.raises(SystemExit) as error:
        Cli().answer_dataset(str(results), str(project / "out"))
    assert error.value.code == 1


def test_answer_single_query(
    project: Path, calls: List[str], capsys: pytest.CaptureFixture[str]
) -> None:
    """A single question prints its answer and sources as JSON."""
    capsys.readouterr()
    Cli().answer("load an adapter", 3)
    printed = json.loads(capsys.readouterr().out)
    assert printed["answer"] == "fake answer"
    assert printed["retrieved_sources"][0]["file_path"] == "data/raw/lora.md"


@pytest.mark.parametrize("query, k", [("", 5), ("  ", 5), ("adapter", 0)])
def test_answer_bad_input_exits_cleanly(
    project: Path, calls: List[str], query: str, k: int
) -> None:
    """Empty query or k <= 0 exits with status 1, no traceback."""
    with pytest.raises(SystemExit) as error:
        Cli().answer(query, k)
    assert error.value.code == 1
    assert calls == []
