"""TF-IDF retrieval implementation."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from retrieval.base import BaseRetriever


class TFIDFRetriever(BaseRetriever):

    def __init__(
        self,
        name: str,
        max_features: int = 10000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 1,
        max_df: float = 1.0,
        top_k: int = 10,
    ) -> None:
        super().__init__(name=name, top_k=top_k)
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.vectorizer: TfidfVectorizer | None = None
        self.document_matrix: np.ndarray | None = None

    def fit(
        self,
        doc_ids: List[str],
        documents: List[str],
    ) -> None:
        self.doc_ids = doc_ids
        self.documents = documents
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
        )
        self.document_matrix = self.vectorizer.fit_transform(self.documents).toarray()

    def build_index(self) -> None:
        self.index_built = True

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> List[Tuple[str, float]]:
        self.validate_index()
        if self.vectorizer is None or self.document_matrix is None:
            raise RuntimeError("TF-IDF model has not been fit.")

        query_vec = self.vectorizer.transform([query]).toarray()
        similarities = linear_kernel(query_vec, self.document_matrix).flatten()
        top_k = top_k or self.top_k
        ranked = sorted(
            enumerate(similarities),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]
        return [
            (self.doc_ids[idx], float(score))
            for idx, score in ranked
        ]

    def save(self, path: str) -> None:
        raise NotImplementedError("TFIDFRetriever save is not implemented.")

    @classmethod
    def load(cls, path: str) -> "TFIDFRetriever":
        raise NotImplementedError("TFIDFRetriever load is not implemented.")
