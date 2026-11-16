import os
from pathlib import Path
from typing import List

from tqdm import tqdm

from src.models import Chunk
from src.parser import chunk_markdown, chunk_python
from src.retriever import Bm25Index


def index_file(index_dir: str) -> Path:
    """Return the path of the index file inside ``index_dir``."""
    return Path(index_dir) / "index.pkl"


def list_files(raw_dir: str) -> List[str]:
    """List the .py, .md and .txt files under ``raw_dir``, sorted.

    Paths are relative to the current directory, with '/' separators, as
    the moulinette expects (e.g. data/raw/vllm-0.10.1/...).
    """
    found: List[str] = []
    for root, _, names in os.walk(raw_dir):
        for name in names:
            if name.endswith((".py", ".md", ".txt")):
                path = os.path.relpath(os.path.join(root, name))
                found.append(path.replace(os.sep, "/"))
    return sorted(found)


def chunk_file(path: str, max_chunk_size: int) -> List[Chunk]:
    """Read one file and split it with the strategy for its type."""
    try:
        with open(path, encoding="utf-8", errors="replace") as file:
            text = file.read()
    except OSError:
        return []
    if path.endswith(".py"):
        return chunk_python(path, text, max_chunk_size, max_chunk_size // 10)
    return chunk_markdown(path, text, max_chunk_size)


def build_index(raw_dir: str, index_dir: str, max_chunk_size: int) -> int:
    """Chunk the corpus, build the BM25 index and save it.

    Returns the number of chunks. Raises ValueError if there is nothing
    to index or ``max_chunk_size`` is not positive.
    """
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be a positive integer")
    files = list_files(raw_dir)
    if not files:
        raise ValueError(f"no .py, .md or .txt files found in {raw_dir}")
    chunks: List[Chunk] = []
    for path in tqdm(files, desc="Chunking", unit="file"):
        chunks.extend(chunk_file(path, max_chunk_size))
    print(f"Building the BM25 index over {len(chunks)} chunks...")
    Bm25Index.build(chunks).save(index_file(index_dir))
    return len(chunks)


def load_index(index_dir: str) -> Bm25Index:
    """Load the saved index, with a clear error if it does not exist."""
    path = index_file(index_dir)
    if not path.is_file():
        raise FileNotFoundError(
            f"no index found in {index_dir}; run the index command first"
        )
    return Bm25Index.load(path)
