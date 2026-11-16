from src.retriever import Bm25Index
from src.models import MinimalSearchResults, MinimalSource
from typing import List


def search_question(
    index: Bm25Index, question_id: str, question: str, k: int
) -> MinimalSearchResults:
    """Search one question and keep only the fields the moulinette expects."""
    sources: List[MinimalSource] = [
        MinimalSource(
            file_path=chunk.file_path,
            first_character_index=chunk.first_character_index,
            last_character_index=chunk.last_character_index,
        )
        for chunk, _ in index.search(question, k)
    ]
    return MinimalSearchResults(
        question_id=question_id, question=question, retrieved_sources=sources
    )
