"""ColBERT-like retrieval implementation."""

from __future__ import annotations

from retrieval.dense import DenseRetriever


class ColBERTRetriever(DenseRetriever):

    def __init__(
        self,
        name: str,
        model_name: str,
        max_doc_len: int = 256,
        max_query_len: int = 64,
        top_k: int = 10,
    ) -> None:
        super().__init__(
            name=name,
            model_name=model_name,
            top_k=top_k,
            normalize=True,
        )
        self.max_doc_len = max_doc_len
        self.max_query_len = max_query_len

    def save(self, path: str) -> None:
        raise NotImplementedError("ColBERTRetriever save is not implemented.")

    @classmethod
    def load(cls, path: str) -> "ColBERTRetriever":
        raise NotImplementedError("ColBERTRetriever load is not implemented.")
