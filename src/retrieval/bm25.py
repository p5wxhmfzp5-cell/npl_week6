"""BM25 retrieval implementation."""

from __future__ import annotations

from typing import List, Tuple
from pathlib import Path

from rank_bm25 import BM25Okapi

from retrieval.base import BaseRetriever


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


class BM25Retriever(BaseRetriever):

    def __init__(
        self,
        name: str,
        k1: float = 1.5,
        b: float = 0.75,
        top_k: int = 10,
    ) -> None:
        super().__init__(name=name, top_k=top_k)
        self.k1 = k1
        self.b = b
        self.bm25: BM25Okapi | None = None

    def fit(
        self,
        doc_ids: List[str],
        documents: List[str],
    ) -> None:
        self.doc_ids = doc_ids
        self.documents = documents
        tokenized = [
            _tokenize(doc)
            for doc in self.documents
        ]
        self.bm25 = BM25Okapi(tokenized)

    def build_index(self) -> None:
        self.index_built = True

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> List[Tuple[str, float]]:
        self.validate_index()
        if self.bm25 is None:
            raise RuntimeError("BM25 index has not been fit.")

        query_tokens = _tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        top_k = top_k or self.top_k
        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]
        return [
            (self.doc_ids[idx], float(score))
            for idx, score in ranked
        ]

    def save(self, path: str) -> None:
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        with open(path_obj / "retriever.txt", "w", encoding="utf-8") as f:
            f.write(f"BM25Retriever name={self.name}\n")

    @classmethod
    def load(cls, path: str) -> "BM25Retriever":
        raise NotImplementedError("BM25Retriever load is not implemented.")
