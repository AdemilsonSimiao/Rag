"""Tests for tokenization and BM25 retrieval in src/retriever.py."""

from pathlib import Path
from typing import List

import pytest

from src.models import Chunk
from src.retriever import Bm25Index, tokenize


def make_chunk(path: str, text: str, start: int = 0) -> Chunk:
    """Build a Chunk for a test."""
    return Chunk(
        file_path=path,
        first_character_index=start,
        last_character_index=start + len(text),
        text=text,
    )


def test_tokenize_lowercases_words() -> None:
    """Words are lowercased and punctuation is dropped."""
    assert tokenize("Load the adapter!") == ["load", "the", "adapter"]


def test_tokenize_keeps_snake_case_whole_and_in_parts() -> None:
    """A snake_case name gives the whole name and each part."""
    assert tokenize("fused_batched_moe") == [
        "fused_batched_moe", "fused", "batched", "moe"
    ]


def test_tokenize_splits_camel_case() -> None:
    """A camelCase name gives the whole name and each part."""
    assert tokenize("LoRAAdapter") == ["loraadapter", "lo", "ra", "adapter"]


def test_tokenize_empty_and_symbols_only() -> None:
    """Text with no word characters gives no tokens."""
    assert tokenize("") == []
    assert tokenize("!!! ??? ---") == []


@pytest.fixture
def index() -> Bm25Index:
    """A tiny index with three chunks about different things."""
    return Bm25Index.build([
        make_chunk("docs/lora.md", "Load a LoRA adapter at runtime."),
        make_chunk("docs/cache.md", "The KV cache stores keys and values."),
        make_chunk("vllm/envs.py", "def get_env(): return environ", 100),
    ])


def test_search_ranks_the_matching_chunk_first(index: Bm25Index) -> None:
    """The chunk with the query words is the top result."""
    results = index.search("how to load an adapter", 3)
    assert results[0][0].file_path == "docs/lora.md"
    assert results[0][1] > 0


def test_search_uses_the_file_path(index: Bm25Index) -> None:
    """A word that is only in the file path still finds the chunk."""
    assert index.search("envs", 1)[0][0].file_path == "vllm/envs.py"


def test_search_returns_scores_in_descending_order(index: Bm25Index) -> None:
    """Results come best first."""
    scores: List[float] = [s for _, s in index.search("cache values adapter", 3)]
    assert scores == sorted(scores, reverse=True)


def test_search_respects_k(index: Bm25Index) -> None:
    """At most k results are returned."""
    assert len(index.search("cache adapter load", 1)) == 1


@pytest.mark.parametrize("query, k", [("adapter", 0), ("adapter", -3),
                                      ("", 5), ("zzzqqq", 5), ("!!!", 5)])
def test_search_degenerate_inputs_return_nothing(
    index: Bm25Index, query: str, k: int
) -> None:
    """k <= 0, an empty query or a query with no match give no results."""
    assert index.search(query, k) == []


def test_search_on_empty_index() -> None:
    """An index with no chunks never fails."""
    assert Bm25Index.build([]).search("anything", 5) == []


def test_save_and_load_roundtrip(index: Bm25Index, tmp_path: Path) -> None:
    """A loaded index gives the same results as the original."""
    path = tmp_path / "sub" / "index.pkl"
    index.save(path)
    loaded = Bm25Index.load(path)
    assert loaded.search("load adapter", 3) == index.search("load adapter", 3)


def test_load_corrupted_file_raises_value_error(tmp_path: Path) -> None:
    """A file that is not an index raises ValueError, not a crash."""
    path = tmp_path / "index.pkl"
    path.write_bytes(b"not a pickle")
    with pytest.raises(ValueError):
        Bm25Index.load(path)


def test_equal_scores_keep_the_chunk_order() -> None:
    """Chunks with the same score come out in index order, every time."""
    same = [make_chunk(f"f{i}.md", "adapter", i * 10) for i in range(5)]
    index = Bm25Index.build(same)
    paths = [chunk.file_path for chunk, _ in index.search("adapter", 5)]
    assert paths == ["f0.md", "f1.md", "f2.md", "f3.md", "f4.md"]
    assert [c.file_path for c, _ in index.search("adapter", 2)] == [
        "f0.md", "f1.md"
    ]
