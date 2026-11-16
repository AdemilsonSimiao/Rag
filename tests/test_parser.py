import pytest

from src.parser import chunk_python


def test_empty_text_gives_no_chunks() -> None:
    """An empty file produces no chunks."""
    assert chunk_python("a.py", "") == []


def test_short_text_is_a_single_chunk() -> None:
    """A file smaller than the window is one chunk covering all of it."""
    chunks = chunk_python("a.py", "x = 1\n", max_chunk_size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0].first_character_index == 0
    assert chunks[0].last_character_index == 6
    assert chunks[0].text == "x = 1\n"


def test_windows_overlap_and_respect_the_limit() -> None:
    """Chunks are at most max_chunk_size long and share the overlap."""
    text = "abcdefghij" * 10  # 100 characters
    chunks = chunk_python("a.py", text, max_chunk_size=40, overlap=10)
    assert [(c.first_character_index, c.last_character_index)
            for c in chunks] == [(0, 40), (30, 70), (60, 100)]
    assert all(len(c.text) <= 40 for c in chunks)
    assert all(
        c.text == text[c.first_character_index:c.last_character_index]
        for c in chunks
    )


def test_file_path_is_kept_verbatim() -> None:
    """The path is copied exactly into every chunk."""
    path = "data/raw/vllm-0.10.1/vllm/envs.py"
    chunks = chunk_python(path, "a" * 5000)
    assert all(c.file_path == path for c in chunks)


@pytest.mark.parametrize(
    "size, overlap", [(0, 0), (-5, 0), (100, -1), (100, 100), (100, 150)]
)
def test_invalid_parameters_raise(size: int, overlap: int) -> None:
    """Bad sizes or overlaps raise ValueError instead of looping forever."""
    with pytest.raises(ValueError):
        chunk_python("a.py", "abc", max_chunk_size=size, overlap=overlap)
