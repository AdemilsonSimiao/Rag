import json
from pathlib import Path
from typing import Dict, List, Optional

from src.cache import JsonCache, file_digest, index_signature
from src.generator import TextGenerator, answer_question, load_generator
from src.indexer import index_file, load_index
from src.models import MinimalSearchResults, MinimalSource
from src.passages import read_passages
from src.retriever import Bm25Index


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


class Searcher:
    """Search questions, using a disk cache and loading the index lazily.

    A question already searched with the same index and the same search
    code is answered from the cache, without loading the index at all.
    """

    def __init__(self, index_dir: str, use_cache: bool = True) -> None:
        """Prepare the searcher; nothing is loaded until it is needed."""
        self.index_dir = index_dir
        self.index: Optional[Bm25Index] = None
        self.cache: Optional[JsonCache] = None
        if use_cache:
            signature = (
                index_signature(index_file(index_dir))
                + file_digest(Path(__file__).with_name("retriever.py"))
            )
            self.cache = JsonCache(
                Path(index_dir) / "query_cache.json", signature
            )

    def search(
        self, question_id: str, question: str, k: int
    ) -> MinimalSearchResults:
        """Return the sources for a question, from the cache if possible."""
        key = JsonCache.make_key(question, str(k))
        if self.cache is not None:
            cached = self.cache.get(key)
            try:
                if cached is not None:
                    return MinimalSearchResults(
                        question_id=question_id,
                        question=question,
                        retrieved_sources=[
                            MinimalSource(**source) for source in cached
                        ],
                    )
            except (TypeError, ValueError):
                pass
        if self.index is None:
            self.index = load_index(self.index_dir)
        result = search_question(self.index, question_id, question, k)
        if self.cache is not None:
            self.cache.set(
                key, [s.model_dump() for s in result.retrieved_sources]
            )
        return result

    def save(self) -> None:
        """Write the cache to disk."""
        if self.cache is not None:
            self.cache.save()


class Answerer:
    """Answer questions, with a disk cache and a lazily loaded model.

    The model is loaded only when an answer is not in the cache and there
    is something to answer from. Each new answer is saved at once, so an
    interrupted run can continue where it stopped.
    """

    def __init__(
        self, model_name: str, cache_dir: str, use_cache: bool = True
    ) -> None:
        """Prepare the answerer; the model is not loaded yet."""
        self.model_name = model_name
        self.model: Optional[TextGenerator] = None
        self.cache: Optional[JsonCache] = None
        if use_cache:
            signature = file_digest(Path(__file__).with_name("generator.py"))
            self.cache = JsonCache(
                Path(cache_dir) / "answer_cache.json", signature
            )

    def generate(self, messages: List[Dict[str, str]]) -> str:
        """Run the model, loading it the first time it is needed."""
        if self.model is None:
            self.model = load_generator(self.model_name)
        return self.model(messages)

    def answer(self, question: str, sources: List[MinimalSource]) -> str:
        """Answer a question from its sources, using the cache if possible."""
        key = JsonCache.make_key(
            self.model_name,
            question,
            json.dumps([s.model_dump() for s in sources]),
        )
        if self.cache is not None:
            cached = self.cache.get(key)
            if isinstance(cached, str):
                return cached
        passages = read_passages(sources)
        text = answer_question(self.generate, question, passages)
        if self.cache is not None and passages:
            self.cache.set(key, text)
            self.cache.save()
        return text
