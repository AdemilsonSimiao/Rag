import re
from typing import List, Tuple
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


def _section_starts(text: str) -> List[int]:
    """Return the offsets where a section starts (0 and each heading)."""
    heading: "re.Pattern[str]" = re.compile(r"#{1,6}[ \t]")
    fences: Tuple[str, str] = ("```", "~~~")
    starts: List[int] = [0]
    in_fence: bool = False
    offset: int = 0
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith(fences):
            in_fence = not in_fence
        elif not in_fence and heading.match(line) and offset > 0:
            starts.append(offset)
        offset += len(line)
    return starts


def _cut_point(text: str, start: int, max_chunk_size: int) -> int:
    """Pick where a piece starting at ''start'' ends, at most max size."""
    limit: int = min(start + max_chunk_size, len(text))
    if limit == len(text):
        return limit
    for separator in ("\n\n", "\n"):
        cut: int = text.rfind(separator, start + 1, limit)
        if cut != -1:
            return cut + len(separator)
    return limit


def chunk_markdown(
    file_path: str, text: str, max_chunk_size: int = 2000
) -> List[Chunk]:
    """Split Markdown or text into sections, cutting long ones by paragraph."""
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be positive")
    bounds: List[int] = _section_starts(text) + [len(text)]
    chunks: List[Chunk] = []
    for section_start, section_end in zip(bounds, bounds[1:]):
        start: int = section_start
        while start < section_end:
            end: int = _cut_point(
                text[:section_end], start, max_chunk_size
            )
            chunks.append(
                Chunk(
                    file_path=file_path,
                    first_character_index=start,
                    last_character_index=end,
                    text=text[start:end],
                )
            )
            start = end
    return chunks
