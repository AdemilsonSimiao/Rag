from pathlib import Path
from tqdm import tqdm
from src.indexer import build_index
from src.models import (
    MinimalAnswer,
    MinimalSearchResults,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer
)
from src.evaluator import evaluate_recall
from src.error import fail
from src.question import Answerer, Searcher
from typing import List


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
        print(f"Ingestion complete! Indeces saved under {index_dir}/")

    def search(
        self,
        query: str,
        k: int = 10,
        index_dir: str = "data/processed",
        use_cache: bool = True,
    ) -> None:
        """Print the top-k sources for a single query."""
        query = str(query)
        if not query.strip():
            fail("the query is empty")
        if k <= 0:
            fail("k must be a positive integer")
        searcher = Searcher(index_dir, use_cache)
        try:
            result = searcher.search("single-query", query, k)
        except (ValueError, OSError) as error:
            fail(str(error))
        searcher.save()
        print(result.model_dump_json(indent=2))

    def search_dataset(
        self,
        dataset_path: str,
        k: int,
        save_directory: str,
        index_dir: str = "data/processed",
        use_cache: bool = True,
    ) -> None:
        """Search a whole dataset and write a StudentSearchResults JSON."""
        if k <= 0:
            fail("k must be a positive integer")
        searcher = Searcher(index_dir, use_cache)
        try:
            dataset = RagDataset.model_validate_json(
                Path(dataset_path).read_text(encoding="utf-8")
            )
            results: List[MinimalSearchResults] = [
                searcher.search(q.question_id, q.question, k)
                for q in tqdm(
                    dataset.rag_questions, desc="Searching", unit="question"
                )
            ]
        except (ValueError, OSError) as error:
            fail(f"cannot read the dataset or the index: {error}")
        searcher.save()
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

    def answer(
        self,
        query: str,
        k: int = 10,
        index_dir: str = "data/processed",
        model_name: str = "Qwen/Qwen3-0.6B",
        use_cache: bool = True,
        cache_dir: str = "data/processed",
    ) -> None:
        """Answer a single query using the retrieved context."""
        query = str(query)
        if not query.strip():
            fail("the query is empty")
        if k <= 0:
            fail("k must be a positive integer")
        searcher = Searcher(index_dir, use_cache)
        try:
            found = searcher.search("single-query", query, k)
        except (ValueError, OSError) as error:
            fail(str(error))
        searcher.save()
        text = Answerer(model_name, cache_dir, use_cache).answer(
            query, found.retrieved_sources
        )
        print(
            MinimalAnswer(
                question_id=found.question_id,
                question=query,
                retrieved_sources=found.retrieved_sources,
                answer=text,
            ).model_dump_json(indent=2)
        )

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_dir: str,
        model_name: str = "Qwen/Qwen3-0.6B",
        use_cache: bool = True,
        cache_dir: str = "data/processed",
    ) -> None:
        """Generate answers for a dataset of search results."""
        try:
            found = StudentSearchResults.model_validate_json(
                Path(student_search_results_path).read_text(encoding="utf-8")
            )
        except (ValueError, OSError) as error:
            fail(f"cannot read the search results: {error}")
        print(f"Loaded {len(found.search_results)} questions")
        answerer  = Answerer(model_name, cache_dir, use_cache)
        answers: List[MinimalAnswer] = []
        for result in tqdm(found.search_results, desc="Answering"):
            text = answerer.answer(result.question, result.retrieved_sources)
            answers.append(
                MinimalAnswer(
                    question_id=result.question_id,
                    question=result.question,
                    retrieved_sources=result.retrieved_sources,
                    answer=text,
                )
            )
        output = Path(save_dir) / Path(student_search_results_path).name
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                StudentSearchResultsAndAnswer(
                    search_results=answers, k=found.k
                ).model_dump_json(indent=2),
                encoding="utf-8"
            )
        except OSError as error:
            fail(f"cannot write {output}: {error}")
        print(
            f"Processed {len(answers)} of {len(answers)} question\n"
            f"Saved student_search_results_and_answer to {output}"
        )

    def evaluate(
        self, student_search_results_path: str, dataset_path: str
    ) -> None:
        """Report recall@k against a ground-truth dataset."""
        try:
            student = StudentSearchResults.model_validate_json(
                Path(student_search_results_path).read_text(encoding="utf-8")
            )
            dataset = RagDataset.model_validate_json(
                Path(dataset_path).read_text(encoding="utf-8")
            )
            recalls = evaluate_recall(student, dataset)
        except (ValueError, OSError) as error:
            fail(f"cannot evaluate search results: {error}")
        for k, recall in recalls.items():
            print(f"Recall@{k}: {recall:.2%}")
