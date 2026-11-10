from typing import List

from src.models import Chunk


def chunk_python(
    file_path: str,
    text: str,
    max_chunk_size: int = 2000,
    overlap: int = 200,
) -> List[Chunk]:
    """Split a Python file into fixed-size windows with overlap."""
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be positive")
    if overlap < 0 or overlap >= max_chunk_size:
        raise ValueError("overlap must be >= 0 and < max_chunk_size")

    chunks: List[Chunk] = []
    step: int = max_chunk_size - overlap
    start: int = 0
    while start < len(text):
        end: int = min(start + max_chunk_size, len(text))
        chunks.append(
            Chunk(
                file_path=file_path,
                first_character_index=start,
                last_character_index=end,
                text=text[start:end],
            )
        )
        if end == len(text):
            break
        start += step
    return chunks
