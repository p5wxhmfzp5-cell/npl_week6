"""Dense retrieval implementation."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from retrieval.base import BaseRetriever


class DenseRetriever(BaseRetriever):

    def __init__(
        self,
        name: str,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 10,
        normalize: bool = True,
    ) -> None:
        super().__init__(name=name, top_k=top_k)
        self.model_name = model_name
        self.normalize = normalize
        self.model = SentenceTransformer(self.model_name)
        self.doc_embeddings: np.ndarray | None = None

    def fit(
        self,
        doc_ids: List[str],
        documents: List[str],
    ) -> None:
        self.doc_ids = doc_ids
        self.documents = documents
        embeddings = self.model.encode(
            self.documents,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        if self.normalize:
            norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norm[norm == 0] = 1.0
            embeddings = embeddings / norm
        self.doc_embeddings = embeddings

    def build_index(self) -> None:
        self.index_built = True

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> List[Tuple[str, float]]:
        self.validate_index()
        if self.doc_embeddings is None:
            raise RuntimeError("Dense retriever has not been fit.")

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        if self.normalize:
            norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
            norm[norm == 0] = 1.0
            query_embedding = query_embedding / norm

        scores = np.dot(self.doc_embeddings, query_embedding.T).flatten()
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
        raise NotImplementedError("DenseRetriever save is not implemented.")

    @classmethod
    def load(cls, path: str) -> "DenseRetriever":
        raise NotImplementedError("DenseRetriever load is not implemented.")
