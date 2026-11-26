"""Small CPU adapter around rank_bm25; no neural model or network access."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from temporalproof.data.timelyqa import Document


@dataclass(frozen=True)
class Hit:
    doc_id: int
    score: float


class BM25Retriever:
    def __init__(
        self,
        documents: Sequence[Document],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ) -> None:
        if not documents or len({doc.global_doc_id for doc in documents}) != len(documents):
            raise ValueError("documents must be nonempty and have unique IDs")
        if not np.isfinite([k1, b, epsilon]).all() or k1 <= 0 or not 0 <= b <= 1 or epsilon < 0:
            raise ValueError("invalid BM25 parameters")
        ordered = sorted(documents, key=lambda doc: doc.global_doc_id)
        corpus = [doc.text.split() for doc in ordered]
        if not any(corpus):
            raise ValueError("corpus must contain at least one token")
        self.doc_ids = np.array([doc.global_doc_id for doc in ordered], dtype=np.int64)
        self.index = BM25Okapi(corpus, k1=k1, b=b, epsilon=epsilon)

    def search(self, query: str, top_k: int) -> list[Hit]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        tokens = query.split()
        if not tokens:
            return []
        scores = self.index.get_scores(tokens)
        if not np.isfinite(scores).all():
            raise ValueError("nonfinite BM25 scores")
        order = np.lexsort((self.doc_ids, -scores))[:top_k]
        return [Hit(int(self.doc_ids[index]), float(scores[index])) for index in order]
