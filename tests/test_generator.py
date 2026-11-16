"""Tests for prompt building and answering in src/generator.py."""

from pathlib import Path
from typing import Dict, List

import pytest

from src.generator import (
    answer_question,
    build_context,
    build_messages,
    clean_answer,
    read_span,
)
from src.models import MinimalSource


def source(path: str, first: int, last: int) -> MinimalSource:
    """Build a MinimalSource for a test."""
    return MinimalSource(
        file_path=path, first_character_index=first, last_character_index=last
    )


def test_read_span_returns_the_requested_characters(tmp_path: Path) -> None:
    """Characters first..last of the file are returned."""
    path = tmp_path / "a.md"
    path.write_text("0123456789")
    assert read_span(str(path), 2, 6) == "2345"


def test_read_span_missing_file_raises_oserror(tmp_path: Path) -> None:
    """A missing file raises OSError, which the CLI skips."""
    with pytest.raises(OSError):
        read_span(str(tmp_path / "nope.md"), 0, 5)


def test_build_context_labels_each_passage() -> None:
    """Each passage gets a number, its file and its character range."""
    context = build_context([(source("docs/a.md", 10, 20), "hello")], 1000)
    assert context.startswith("[1] docs/a.md (characters 10-20)\n")
    assert context.endswith("hello")


def test_build_context_keeps_the_order() -> None:
    """Passages appear in the order given (best first)."""
    context = build_context(
        [(source("a", 0, 1), "AAA"), (source("b", 0, 1), "BBB")], 1000
    )
    assert context.index("AAA") < context.index("BBB")


def test_build_context_respects_the_character_budget() -> None:
    """The context never exceeds max_chars, and later passages are dropped."""
    passages = [(source("a.md", 0, 500), "x" * 500) for _ in range(5)]
    context = build_context(passages, 700)
    assert len(context) <= 700
    assert "[2]" in context
    assert "[3]" not in context


def test_build_context_with_no_passages_is_empty() -> None:
    """No passages give an empty context."""
    assert build_context([], 1000) == ""


def test_build_messages_has_system_and_user_roles() -> None:
    """The prompt carries the context and the question."""
    messages = build_messages("What is X?", "some context")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "some context" in messages[1]["content"]
    assert "What is X?" in messages[1]["content"]


@pytest.mark.parametrize("raw, expected", [
    ("  plain answer \n", "plain answer"),
    ("<think>hidden</think>The answer.", "The answer."),
    ("<think>\nmulti\nline\n</think>\n\nDone", "Done"),
    ("Answer<think>unfinished", "Answer"),
    ("<think>only thinking", ""),
])
def test_clean_answer(raw: str, expected: str) -> None:
    """Thinking blocks are removed and the text is trimmed."""
    assert clean_answer(raw) == expected


def test_answer_question_uses_the_model_reply() -> None:
    """The reply of the generator is returned, cleaned."""
    seen: List[List[Dict[str, str]]] = []

    def fake(messages: List[Dict[str, str]]) -> str:
        seen.append(messages)
        return "<think>x</think> POST /v1/load_lora_adapter "

    reply = answer_question(
        fake, "Which endpoint?", [(source("a.md", 0, 5), "text")]
    )
    assert reply == "POST /v1/load_lora_adapter"
    assert "text" in seen[0][1]["content"]


def test_answer_question_without_passages_does_not_call_the_model() -> None:
    """With nothing retrieved, the model is not asked to invent an answer."""
    def fake(messages: List[Dict[str, str]]) -> str:
        raise AssertionError("the model must not be called")

    assert "No relevant sources" in answer_question(fake, "Q?", [])


def test_answer_question_with_empty_reply_gives_a_fallback() -> None:
    """An empty model reply is replaced by a short message."""
    def fake(messages: List[Dict[str, str]]) -> str:
        return "<think>only thoughts"

    reply = answer_question(fake, "Q?", [(source("a", 0, 1), "t")])
    assert reply == "The model did not produce an answer."
