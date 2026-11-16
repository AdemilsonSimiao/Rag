from pathlib import Path
from tqdm import tqdm
from src.indexer import build_index, load_index
from src.models import (
    MinimalAnswer,
    MinimalSearchResults,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from src.evaluator import evaluate_recall
from src.error import fail
from src.generator import answer_question, load_generator
from src.question import search_question
from typing import List
from src.passages import read_passages


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

    def answer(
        self,
        query: str,
        k: int = 10,
        index_dir: str = "data/processed",
        model_name: str = "Qwen/Qwen3-0.6B",
    ) -> None:
        """Answer a single query using the retrieved context."""
        query = str(query)
        if not query.strip():
            fail("the query is empty")
        if k <= 0:
            fail("k must be a positive integer")
        try:
            index = load_index(index_dir)
        except (ValueError, OSError) as error:
            fail(str(error))
        found = search_question(index, "single-query", query, k)
        generate = load_generator(model_name)
        text = answer_question(
            generate, query, read_passages(found.retrieved_sources)
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
        save_directory: str,
        model_name: str = "Qwen/Qwen3-0.6B",
    ) -> None:
        """Generate answers for a dataset of search results."""
        try:
            found = StudentSearchResults.model_validate_json(
                Path(student_search_results_path).read_text(encoding="utf-8")
            )
        except (ValueError, OSError) as error:
            fail(f"cannot read the search results: {error}")
        print(f"Loaded {len(found.search_results)} questions")
        generate = load_generator(model_name)
        answers: List[MinimalAnswer] = []
        for result in tqdm(found.search_results, desc="Answering"):
            text = answer_question(
                generate,
                result.question,
                read_passages(result.retrieved_sources),
            )
            answers.append(
                MinimalAnswer(
                    question_id=result.question_id,
                    question=result.question,
                    retrieved_sources=result.retrieved_sources,
                    answer=text,
                )
            )
        output = Path(save_directory) / Path(student_search_results_path).name
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                StudentSearchResultsAndAnswer(
                    search_results=answers, k=found.k
                ).model_dump_json(indent=2),
                encoding="utf-8",
            )
        except OSError as error:
            fail(f"cannot write {output}: {error}")
        print(
            f"Processed {len(answers)} of {len(answers)} questions\n"
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
