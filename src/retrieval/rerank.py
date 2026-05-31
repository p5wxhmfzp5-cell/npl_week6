"""Cross-encoder reranking retrieval implementation."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from retrieval.base import BaseRetriever
from retrieval.bm25 import BM25Retriever
from retrieval.tfidf import TFIDFRetriever


class CrossEncoderRetriever(BaseRetriever):

    def __init__(
        self,
        name: str,
        first_stage: str = "bm25",
        reranker_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        candidate_pool: int = 100,
        top_k: int = 10,
    ) -> None:
        super().__init__(name=name, top_k=top_k)
        self.first_stage = first_stage
        self.reranker_model = reranker_model
        self.candidate_pool = candidate_pool
        self.first_retriever: BaseRetriever | None = None
        self.reranker = SentenceTransformer(self.reranker_model)
        self.document_embeddings: np.ndarray | None = None

    def _build_first_stage(self) -> BaseRetriever:
        if self.first_stage == "bm25":
            return BM25Retriever(name=f"{self.name}_bm25")
        if self.first_stage == "tfidf":
            return TFIDFRetriever(name=f"{self.name}_tfidf")
        raise ValueError(
            f"Unsupported first stage: {self.first_stage}"
        )

    def fit(
        self,
        doc_ids: List[str],
        documents: List[str],
    ) -> None:
        self.doc_ids = doc_ids
        self.documents = documents
        self.first_retriever = self._build_first_stage()
        self.first_retriever.fit(doc_ids, documents)
        self.first_retriever.build_index()
        self.document_embeddings = self.reranker.encode(
            self.documents,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def build_index(self) -> None:
        self.index_built = True

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> List[Tuple[str, float]]:
        self.validate_index()
        if self.first_retriever is None or self.document_embeddings is None:
            raise RuntimeError("Cross encoder retriever has not been fit.")

        pool = self.first_retriever.retrieve(
            query=query,
            top_k=self.candidate_pool,
        )
        if not pool:
            return []

        candidate_ids = [doc_id for doc_id, _ in pool]
        candidate_texts = [
            self.documents[self.doc_ids.index(doc_id)]
            for doc_id in candidate_ids
        ]
        query_embedding = self.reranker.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        scores = np.dot(
            self.document_embeddings[
                [self.doc_ids.index(doc_id) for doc_id in candidate_ids]
            ],
            query_embedding.T,
        ).flatten()
        top_k = top_k or self.top_k
        ranked = sorted(
            zip(candidate_ids, scores),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]
        return [
            (doc_id, float(score))
            for doc_id, score in ranked
        ]

    def save(self, path: str) -> None:
        raise NotImplementedError("CrossEncoderRetriever save is not implemented.")

    @classmethod
    def load(cls, path: str) -> "CrossEncoderRetriever":
        raise NotImplementedError("CrossEncoderRetriever load is not implemented.")
