import math
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt

from src.models import Chunk


def tokenize(text: str) -> List[str]:
    """Lowercase words; also split snake_case and camelCase into parts."""
    tokens: List[str] = []
    for word in re.findall(r"[A-Za-z0-9_]+", text):
        tokens.append(word.lower())
        pattern = r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+"
        parts = re.findall(pattern, word)
        if len(parts) > 1:
            tokens.extend(part.lower() for part in parts)
    return tokens


class Bm25Index:
    """BM25 index over chunks, with the term scores computed at build time."""

    def __init__(
        self,
        chunks: List[Chunk],
        postings: Dict[
            str, Tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]
        ],
    ) -> None:
        """Store the chunks and, per term, the chunks it appears in."""
        self.chunks: List[Chunk] = chunks
        self.postings = postings

    @classmethod
    def build(
        cls, chunks: List[Chunk], k1: float = 1.5, b: float = 0.75
    ) -> "Bm25Index":
        """Tokenize every chunk (path included) and precompute BM25 scores."""
        doc_ids: Dict[str, List[int]] = {}
        freqs: Dict[str, List[int]] = {}
        lengths: List[int] = []
        for doc_id, chunk in enumerate(chunks):
            tokens = tokenize(chunk.file_path + "\n" + chunk.text)
            lengths.append(len(tokens))
            for term, freq in Counter(tokens).items():
                doc_ids.setdefault(term, []).append(doc_id)
                freqs.setdefault(term, []).append(freq)

        doc_len = np.array(lengths, dtype=np.float32)
        avg_len = float(doc_len.mean()) if lengths else 1.0
        postings: Dict[
            str, Tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]
        ] = {}
        for term, ids in doc_ids.items():
            tf = np.array(freqs[term], dtype=np.float32)
            df = len(ids)
            idf = math.log(1 + (len(chunks) - df + 0.5) / (df + 0.5))
            norm = k1 * (1 - b + b * doc_len[ids] / avg_len)
            weights = (idf * tf * (k1 + 1) / (tf + norm)).astype(np.float32)
            postings[term] = (np.array(ids, dtype=np.int32), weights)
        return cls(chunks, postings)

    def search(self, query: str, k: int) -> List[Tuple[Chunk, float]]:
        """Return the top-k chunks for the query, best first, with scores."""
        if k <= 0 or not self.chunks:
            return []
        scores = np.zeros(len(self.chunks), dtype=np.float32)
        for term in set(tokenize(query)):
            posting = self.postings.get(term)
            if posting is not None:
                scores[posting[0]] += posting[1]
        k = min(k, len(self.chunks))
        best = np.argpartition(-scores, k - 1)[:k]
        best = best[np.argsort(-scores[best], kind="stable")]
        return [
            (self.chunks[i], float(scores[i])) for i in best if scores[i] > 0
        ]

    def save(self, path: Path) -> None:
        """Write the index to a file, creating the folder if needed."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "Bm25Index":
        """Read an index written by ``save``."""
        try:
            with open(path, "rb") as file:
                index = pickle.load(file)
        except (pickle.UnpicklingError, EOFError, AttributeError) as error:
            raise ValueError(f"index file {path} is corrupted") from error
        if not isinstance(index, cls):
            raise ValueError(f"index file {path} is not a BM25 index")
        return index
