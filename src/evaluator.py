from typing import Dict, Tuple

from src.models import AnsweredQuestion, RagDataset, StudentSearchResults


def _overlaps(
    retrieved: Tuple[str, int, int],
    expected: Tuple[str, int, int],
) -> bool:
    """True if same file and the character ranges have IoU >= 0.05."""
    retrieved_path, retrieved_start, retrieved_end = retrieved
    expected_path, expected_start, expected_end = expected

    overlap = min(retrieved_end, expected_end) - max(
        retrieved_start, expected_start
    )
    if retrieved_path != expected_path or overlap <= 0:
        return False
    union = (
        (retrieved_end - retrieved_start)
        + (expected_end - expected_start)
        - overlap
    )
    return overlap / union >= 0.05


def evaluate_recall(
    student: StudentSearchResults,
    dataset: RagDataset,
    ks: Tuple[int, ...] = (1, 3, 5, 10),
) -> Dict[int, float]:
    """Return macro-averaged Recall@k for answered questions."""
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("all k values must be positive")

    results_by_id = {
        result.question_id: result
        for result in student.search_results
    }

    questions = [
        question
        for question in dataset.rag_questions
        if isinstance(question, AnsweredQuestion) and question.sources
    ]
    if not questions:
        raise ValueError("the dataset has no answered questions with sources")

    recalls: Dict[int, float] = {}

    for k in ks:
        question_recalls = []

        for question in questions:
            result = results_by_id.get(question.question_id)
            retrieved = result.retrieved_sources[:k] if result else []

            expected_sources = [
                (
                    source.file_path,
                    source.first_character_index,
                    source.last_character_index,
                )
                for source in question.sources
            ]
            retrieved_sources = [
                (
                    source.file_path,
                    source.first_character_index,
                    source.last_character_index,
                )
                for source in retrieved
            ]

            found = sum(
                any(_overlaps(candidate, expected) for candidate in retrieved_sources)
                for expected in expected_sources
            )
            question_recalls.append(found / len(expected_sources))

        recalls[k] = sum(question_recalls) / len(question_recalls)

    return recalls
