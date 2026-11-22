import math
import pickle
import re
import numpy as np
import numpy.typing as npt
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple
from src.models import Chunk


def tokenize(text: str) -> List[str]:
    """Lowercase words; also snake_case and camelCase into parts."""
    tokens: List[str] = []
    for word in re.findall(r"[A-Za-z0-9_]+", text):
        tokens.append(word.lower())
        pattern = r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+"
        parts = re.findall(pattern, word)
        if len(parts) > 1:
            tokens.extend(part.lower() for part in parts)
    return tokens


class Bm25Index:
    """Bm25 index over chunks, with the term scores computed at build time."""
    def __init__(
        self,
        chunks: List[Chunk],
        term_ranges: Dict[str, Tuple[int, int]],
        doc_ids: npt.NDArray[np.int32],
        weights: npt.NDArray[np.float32],
    ) -> None:
        """Store the chunks and the postings of all terms in two big arrays."""
        self.chunks: List[Chunk] = chunks
        self.term_ranges = term_ranges
        self.doc_ids = doc_ids
        self.weights = weights

    @classmethod
    def build(
        cls, chunks: List[Chunk], k1: float = 1.5, b: float = 0.75
    ) -> "Bm25Index":
        """Tokenize every chunk (path include) and precompute BM25 scores."""
        term_docs: Dict[str, List[int]] = {}
        freqs: Dict[str, List[int]] = {}
        lengths: List[int] = []
        for doc_id, chunk in enumerate(chunks):
            tokens = tokenize(chunk.file_path + "\n" + chunk.text)
            lengths.append(len(tokens))
            for term, freq in Counter(tokens).items():
                term_docs.setdefault(term, []).append(doc_id)
                freqs.setdefault(term, []).append(freq)
        doc_len = np.array(lengths, dtype=np.float32)
        avg_len = float(doc_len.mean()) if lengths else 1.0
        term_ranges: Dict[str, Tuple[int, int]] = {}
        id_parts: List[npt.NDArray[np.int32]] = []
        weight_parts: List[npt.NDArray[np.float32]] = []
        position = 0
        for term, ids in term_docs.items():
            tf = np.array(freqs[term], dtype=np.float32)
            df = len(ids)
            idf = math.log(1 + (len(chunks) - df + 0.5) / (df + 0.5))
            norm = k1 * (1 - b + b * doc_len[ids] / avg_len)
            id_parts.append(np.array(ids, dtype=np.int32))
            weight_parts.append(
                (idf * tf * (k1 + 1) / (tf + norm)).astype(np.float32)
            )
            term_ranges[term] = (position, position + df)
            position += df
        if not id_parts:
            return cls(
                chunks,
                term_ranges,
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.float32),
            )
        return cls(
            chunks,
            term_ranges,
            np.concatenate(id_parts),
            np.concatenate(weight_parts),
        )

    def search(self, query: str, k: int) -> List[Tuple[Chunk, float]]:
        """"Return the top-k chunks for the query, best first, with scores."""
        if k <= 0 or not self.chunks:
            return []
        scores = np.zeros(len(self.chunks), dtype=np.float32)
        for term in set(tokenize(query)):
            found = self.term_ranges.get(term)
            if found is not None:
                start, end = found
                scores[self.doc_ids[start:end]] += self.weights[start:end]
        best = np.lexsort((np.arange(len(scores)), -scores))[:k]
        return [
            (self.chunks[i], float(scores[i])) for i in best if scores[i] > 0
        ]

    def save(self, path: Path) -> None:
        """Write the index to a file, creating the folder is needed."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "Bm25Index":
        """"Read an index written by ''save''."""
        try:
            with open(path, "rb") as file:
                index = pickle.load(file)
        except (pickle.UnpicklingError, EOFError, ArithmeticError) as error:
            raise ValueError(
                f"index file {path} is corrupted or from an older version;"
                "run the index command again"
            ) from error
        if not isinstance(index, cls) or not hasattr(index, "term_ranges"):
            raise ValueError(
                f"index file {path} is not valid BM25 index; "
                "run the index command again"
            )
        return index
