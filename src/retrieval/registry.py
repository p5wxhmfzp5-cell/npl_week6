"""
Retriever Registry

Purpose
-------
Central factory + registry for all benchmark retrievers.

Supports:

- BM25
- TF-IDF
- Dense MiniLM
- Dense M3
- Hybrid RRF
- ColBERT CPU
- Cross-Encoder ReRank
- HyDE

Used by:

- cli.py
- BenchmarkRunner
- experiments
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Type

from retrieval.base import BaseRetriever
from utils.config import (
    load_yaml_config,
)

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    .parent
)


############################################################
# IMPORT RETRIEVERS
############################################################

# imports are explicit for clarity/reproducibility

from retrieval.bm25 import (
    BM25Retriever,
)

from retrieval.tfidf import (
    TFIDFRetriever,
)

from retrieval.dense import (
    DenseRetriever,
)

from retrieval.hybrid_rrf import (
    HybridRRFRetriever,
)

from retrieval.colbert_cpu import (
    ColBERTRetriever,
)

from retrieval.rerank import (
    CrossEncoderRetriever,
)

from retrieval.hyde import (
    HyDERetriever,
)


############################################################
# REGISTRY
############################################################

RETRIEVER_REGISTRY: Dict[
    str,
    Type[BaseRetriever],
] = {

    "bm25": BM25Retriever,

    "tfidf": TFIDFRetriever,

    "dense_minilm": DenseRetriever,

    "dense_m3": DenseRetriever,

    "hybrid_rrf": HybridRRFRetriever,

    "colbert_cpu": ColBERTRetriever,

    "cross_encoder_rerank":
        CrossEncoderRetriever,

    "hyde": HyDERetriever,
}


############################################################
# CONFIG RESOLUTION
############################################################

def model_config_path(
    method_name: str,
) -> Path:
    """
    Resolve model config YAML.
    """

    return (
        PROJECT_ROOT
        / "configs"
        / "models"
        / f"{method_name}.yaml"
    )


############################################################
# FACTORY
############################################################

def get_retriever(
    method_name: str,
) -> BaseRetriever:
    """
    Factory constructor.

    Parameters
    ----------
    method_name:
        benchmark method name

    Returns
    -------
    instantiated retriever
    """

    method_name = (
        method_name
        .strip()
        .lower()
    )

    if method_name not in (
        RETRIEVER_REGISTRY
    ):

        available = list(
            RETRIEVER_REGISTRY.keys()
        )

        raise ValueError(
            f"Unknown retriever: "
            f"{method_name}\n"
            f"Available: {available}"
        )

    cfg_path = model_config_path(
        method_name
    )

    cfg = load_yaml_config(
        cfg_path
    )

    retriever_cls = (
        RETRIEVER_REGISTRY[
            method_name
        ]
    )

    ########################################################
    # Dense specialization
    ########################################################

    if method_name in (
        "dense_minilm",
        "dense_m3",
    ):

        model_name = (
            cfg.encoder
            .model_name
        )

        return retriever_cls(
            name=cfg.name,
            model_name=model_name,
            top_k=10,
            normalize=(
                cfg
                .normalize_embeddings
            ),
        )

    ########################################################
    # BM25
    ########################################################

    if method_name == "bm25":

        return retriever_cls(

            name=cfg.name,

            k1=cfg.parameters.k1,

            b=cfg.parameters.b,

            top_k=10,
        )

    ########################################################
    # TFIDF
    ########################################################

    if method_name == "tfidf":

        return retriever_cls(

            name=cfg.name,

            max_features=(
                cfg.parameters
                .max_features
            ),

            ngram_range=tuple(
                cfg.parameters
                .ngram_range
            ),

            min_df=(
                cfg.parameters
                .min_df
            ),

            max_df=(
                cfg.parameters
                .max_df
            ),
        )

    ########################################################
    # Hybrid
    ########################################################

    if method_name == "hybrid_rrf":

        return retriever_cls(
            name=cfg.name,

            sparse_method=(
                cfg.components
                .sparse
            ),

            dense_method=(
                cfg.components
                .dense
            ),

            rrf_k=(
                cfg.fusion.k
            ),
        )

    ########################################################
    # ColBERT
    ########################################################

    if method_name == "colbert_cpu":

        return retriever_cls(

            name=cfg.name,

            model_name=(
                cfg.encoder
                .model_name
            ),

            max_doc_len=(
                cfg
                .max_document_length
            ),

            max_query_len=(
                cfg
                .max_query_length
            ),
        )

    ########################################################
    # ReRanker
    ########################################################

    if method_name == (
        "cross_encoder_rerank"
    ):

        return retriever_cls(

            name=cfg.name,

            first_stage=(
                cfg.pipeline
                .first_stage
            ),

            reranker_model=(
                cfg.pipeline
                .reranker_model
            ),

            candidate_pool=(
                cfg.retrieval
                .candidate_pool
            ),
        )

    ########################################################
    # HyDE
    ########################################################

    if method_name == "hyde":

        return retriever_cls(

            name=cfg.name,

            generator_model=(
                cfg.generator
                .model_name
            ),

            encoder_model=(
                cfg.encoder
                .model_name
            ),

            max_tokens=(
                cfg.generator
                .max_new_tokens
            ),
        )

    ########################################################
    # Fallback
    ########################################################

    return retriever_cls(
        name=cfg.name
    )


############################################################
# LISTING
############################################################

def available_retrievers():
    """
    List supported methods.
    """

    return sorted(
        RETRIEVER_REGISTRY.keys()
    )


############################################################
# REGISTRATION EXTENSION
############################################################

def register_retriever(
    name: str,
    cls: Type[BaseRetriever],
):
    """
    Runtime plugin registration.
    """

    if not issubclass(
        cls,
        BaseRetriever,
    ):

        raise TypeError(
            "Retriever must inherit "
            "BaseRetriever."
        )

    RETRIEVER_REGISTRY[
        name
    ] = cls


############################################################
# EXPORTS
############################################################

__all__ = [

    "get_retriever",

    "available_retrievers",

    "register_retriever",
]