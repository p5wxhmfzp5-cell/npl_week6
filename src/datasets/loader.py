"""Dataset loader."""

from __future__ import annotations

from typing import Dict

from .msmarco import load_msmarco, download_msmarco
from .scifact import load_scifact, download_scifact


def load_dataset(dataset: str) -> Dict[str, list]:
    dataset = dataset.strip().lower()

    if dataset == "msmarco":
        return load_msmarco()

    if dataset == "scifact":
        return load_scifact()

    raise ValueError(
        f"Unsupported dataset: {dataset}. "
        "Supported values are: msmarco, scifact."
    )
