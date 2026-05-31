"""Hybrid RRF retrieval implementation."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from retrieval.base import BaseRetriever
from retrieval.bm25 import BM25Retriever
from retrieval.tfidf import TFIDFRetriever
from retrieval.dense import DenseRetriever


class HybridRRFRetriever(BaseRetriever):

    def __init__(
        self,
        name: str,
        sparse_method: str = "bm25",
        dense_method: str = "dense_minilm",
        rrf_k: int = 60,
        top_k: int = 10,
    ) -> None:
        super().__init__(name=name, top_k=top_k)
        self.sparse_method = sparse_method
        self.dense_method = dense_method
        self.rrf_k = rrf_k
        self.sparse_retriever: BaseRetriever | None = None
        self.dense_retriever: DenseRetriever | None = None

    def _build_sparse(self) -> BaseRetriever:
        if self.sparse_method == "bm25":
            return BM25Retriever(name=f"{self.name}_sparse")
        if self.sparse_method == "tfidf":
            return TFIDFRetriever(name=f"{self.name}_sparse")
        raise ValueError(
            f"Unsupported sparse method: {self.sparse_method}"
        )

    def _build_dense(self) -> DenseRetriever:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        if self.dense_method == "dense_m3":
            model_name = "sentence-transformers/all-mpnet-base-v2"
        return DenseRetriever(
            name=f"{self.name}_dense",
            model_name=model_name,
            normalize=True,
        )

    def fit(
        self,
        doc_ids: List[str],
        documents: List[str],
    ) -> None:
        self.doc_ids = doc_ids
        self.documents = documents
        self.sparse_retriever = self._build_sparse()
        self.sparse_retriever.fit(doc_ids, documents)
        self.sparse_retriever.build_index()
        self.dense_retriever = self._build_dense()
        self.dense_retriever.fit(doc_ids, documents)
        self.dense_retriever.build_index()

    def build_index(self) -> None:
        self.index_built = True

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> List[Tuple[str, float]]:
        self.validate_index()
        if self.sparse_retriever is None or self.dense_retriever is None:
            raise RuntimeError("Hybrid retriever has not been fit.")

        top_k = top_k or self.top_k
        sparse_results = self.sparse_retriever.retrieve(
            query=query,
            top_k=self.rrf_k,
        )
        dense_results = self.dense_retriever.retrieve(
            query=query,
            top_k=self.rrf_k,
        )

        scores: dict[str, float] = {}
        for rank, (doc_id, score) in enumerate(sparse_results, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rank + self.rrf_k)
        for rank, (doc_id, score) in enumerate(dense_results, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rank + self.rrf_k)

        ranked = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]
        return [(doc_id, float(score)) for doc_id, score in ranked]

    def save(self, path: str) -> None:
        raise NotImplementedError("HybridRRFRetriever save is not implemented.")

    @classmethod
    def load(cls, path: str) -> "HybridRRFRetriever":
        raise NotImplementedError("HybridRRFRetriever load is not implemented.")
