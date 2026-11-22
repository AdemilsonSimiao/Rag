from typing import List, Tuple
from src.generator import read_span
from src.models import MinimalSource


def read_passages(
    sources: List[MinimalSource],
) -> List[Tuple[MinimalSource, str]]:
    """Read the text of each source, skipping files that cannot be read."""
    passages: List[Tuple[MinimalSource, str]] = []
    for source in sources:
        try:
            text = read_span(
                source.file_path,
                source.first_character_index,
                source.last_character_index,
            )
        except OSError:
            continue
        passages.append((source, text))
    return passages
