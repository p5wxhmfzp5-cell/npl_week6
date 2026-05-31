"""
Base Retriever Interface

All retrieval methods in the benchmark inherit from this class.

Supported methods:

- BM25
- TF-IDF
- Dense Retrieval
- Hybrid RRF
- ColBERT
- Cross-Encoder Re-Ranking
- HyDE
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any

import time


class BaseRetriever(ABC):
    """
    Unified retriever abstraction.

    Every retrieval implementation must expose:

        fit()
        build_index()
        retrieve()
        save()
        load()

    This guarantees identical benchmarking behavior.
    """

    def __init__(
        self,
        name: str,
        top_k: int = 10,
    ) -> None:

        self.name = name
        self.top_k = top_k

        self.index_built = False

        self.doc_ids: List[str] = []
        self.documents: List[str] = []

    @abstractmethod
    def fit(
        self,
        doc_ids: List[str],
        documents: List[str],
    ) -> None:
        """
        Train / initialize retriever.

        Sparse methods:
            tokenize, vocab construction.

        Dense methods:
            encoder loading.

        Parameters
        ----------
        doc_ids : List[str]

        documents : List[str]
        """
        raise NotImplementedError

    @abstractmethod
    def build_index(self) -> None:
        """
        Build retrieval index.

        Examples:

        BM25:
            tokenized corpus

        Dense:
            FAISS index

        ColBERT:
            late-interaction index
        """
        raise NotImplementedError

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> List[Tuple[str, float]]:
        """
        Retrieve ranked documents.

        Returns
        -------
        List[
            (doc_id, score)
        ]
        """
        raise NotImplementedError

    @abstractmethod
    def save(
        self,
        path: str,
    ) -> None:
        """
        Persist retriever artifacts.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(
        cls,
        path: str,
    ) -> "BaseRetriever":
        """
        Restore retriever from disk.
        """
        raise NotImplementedError

    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int | None = None,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Batched retrieval.

        Default implementation loops.

        Dense retrievers can override for speed.
        """

        outputs = {}

        for q in queries:
            outputs[q] = self.retrieve(
                query=q,
                top_k=top_k,
            )

        return outputs

    def benchmark_latency(
        self,
        queries: List[str],
        top_k: int = 10,
    ) -> Dict[str, float]:
        """
        Measure average retrieval latency.
        """

        start = time.perf_counter()

        for q in queries:

            self.retrieve(
                q,
                top_k=top_k,
            )

        elapsed = time.perf_counter() - start

        avg_latency_ms = (
            elapsed / len(queries)
        ) * 1000

        throughput = (
            len(queries) / elapsed
        )

        return {
            "avg_latency_ms": round(
                avg_latency_ms,
                3,
            ),
            "throughput_qps": round(
                throughput,
                3,
            ),
        }

    def info(self) -> Dict[str, Any]:
        """
        Metadata used by benchmark runner.
        """

        return {
            "name": self.name,
            "top_k": self.top_k,
            "index_built": self.index_built,
            "n_docs": len(self.documents),
        }

    def validate_index(self) -> None:
        """
        Ensure index exists before retrieval.
        """

        if not self.index_built:

            raise RuntimeError(
                f"{self.name} index not built."
            )

    def __repr__(self) -> str:

        return (
            f"{self.__class__.__name__}("
            f"name={self.name}, "
            f"top_k={self.top_k}, "
            f"docs={len(self.documents)})"
        )