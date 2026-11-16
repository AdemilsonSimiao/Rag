import sys
from pathlib import Path
from typing import List, NoReturn

import fire
from tqdm import tqdm

from src.indexer import build_index, load_index
from src.models import (
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)
from src.retriever import Bm25Index


def fail(message: str) -> NoReturn:
    """Print an error message and exit with status 1, without a traceback."""
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(1)


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


class Cli:
    """RAG commands exposed through Python Fire."""

    def index(
        self,
        max_chunk_size: int = 2000,
        raw_dir: str = "data/raw",
        index_dir: str = "data/processed",
    ) -> None:
        """Ingest raw_dir and build the index under index_dir."""
        try:
            build_index(raw_dir, index_dir, max_chunk_size)
        except (ValueError, OSError) as error:
            fail(str(error))
        print(f"Ingestion complete! Indices saved under {index_dir}/")

    def search(
        self, query: str, k: int = 10, index_dir: str = "data/processed"
    ) -> None:
        """Print the top-k sources for a single query."""
        query = str(query)
        if not query.strip():
            fail("the query is empty")
        if k <= 0:
            fail("k must be a positive integer")
        try:
            index = load_index(index_dir)
        except (ValueError, OSError) as error:
            fail(str(error))
        result = search_question(index, "single-query", query, k)
        print(result.model_dump_json(indent=2))

    def search_dataset(
        self,
        dataset_path: str,
        k: int,
        save_directory: str,
        index_dir: str = "data/processed",
    ) -> None:
        """Search a whole dataset and write a StudentSearchResults JSON."""
        if k <= 0:
            fail("k must be a positive integer")
        try:
            index = load_index(index_dir)
            dataset = RagDataset.model_validate_json(
                Path(dataset_path).read_text(encoding="utf-8")
            )
        except (ValueError, OSError) as error:
            fail(f"cannot read the dataset or the index: {error}")
        results: List[MinimalSearchResults] = [
            search_question(index, q.question_id, q.question, k)
            for q in tqdm(
                dataset.rag_questions, desc="Searching", unit="question"
            )
        ]
        output = Path(save_directory) / Path(dataset_path).name
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                StudentSearchResults(
                    search_results=results, k=k
                ).model_dump_json(indent=2),
                encoding="utf-8",
            )
        except OSError as error:
            fail(f"cannot write {output}: {error}")
        print(f"Saved student_search_results to {output}")

    def answer(self, query: str, k: int = 10) -> None:
        """Answer a single query using the retrieved context."""
        fail("answer is not implemented yet")

    def answer_dataset(
        self, student_search_results_path: str, save_directory: str
    ) -> None:
        """Generate answers for a dataset of search results."""
        fail("answer_dataset is not implemented yet")

    def evaluate(
        self, student_search_results_path: str, dataset_path: str
    ) -> None:
        """Report recall@k against a ground-truth dataset."""
        fail("evaluate is not implemented yet")


if __name__ == "__main__":
    fire.Fire(Cli)
