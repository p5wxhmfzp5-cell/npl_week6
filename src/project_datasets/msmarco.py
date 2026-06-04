"""MS MARCO dataset utilities."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from utils.config import (
    dataset_config_path,
    load_yaml_config,
)


def download_msmarco() -> None:
    raise NotImplementedError(
        "MS MARCO download is not implemented in this repository skeleton. "
        "Add a downloader or populate data/raw/msmarco manually."
    )


def load_msmarco() -> Dict[str, list]:
    cfg = load_yaml_config(
        dataset_config_path("msmarco")
    )

    raw_dir = Path(cfg.storage.raw_dir)

    if not raw_dir.exists() or not raw_dir.is_dir():
        return _sample_msmarco()

    corpus_file = _find_file(
        raw_dir,
        [
            "corpus.tsv",
            "collection.tsv",
            "collection.tsv.gz",
            "corpus.txt",
            "msmarco-docs.tsv",
        ],
    )
    query_file = _find_file(
        raw_dir,
        [
            "queries.tsv",
            "queries.txt",
            "queries.dev.tsv",
            "queries.dev.small.tsv",
            "queries.small.tsv",
        ],
    )
    qrels_file = _find_file(
        raw_dir,
        [
            "qrels.dev.small.tsv",
            "qrels.dev.tsv",
            "qrels.tsv",
            "qrels.train.tsv",
        ],
    )

    if corpus_file is None:
        raise FileNotFoundError(
            f"Could not find MS MARCO corpus file in {raw_dir}."
        )

    if query_file is None:
        raise FileNotFoundError(
            f"Could not find MS MARCO query file in {raw_dir}."
        )

    if qrels_file is None:
        raise FileNotFoundError(
            f"Could not find MS MARCO qrels file in {raw_dir}."
        )

    doc_ids, documents = _load_pairs(corpus_file)
    query_ids, queries = _load_pairs(query_file)
    qrels = _load_qrels(qrels_file)

    return {
        "doc_ids": doc_ids,
        "documents": documents,
        "query_ids": query_ids,
        "queries": queries,
        "qrels": qrels,
    }


def _sample_msmarco() -> Dict[str, list]:
    return {
        "doc_ids": ["D1", "D2", "D3"],
        "documents": [
            "The quick brown fox jumps over the lazy dog.",
            "MS MARCO is a large passage retrieval dataset.",
            "This sample passage is useful for local testing.",
        ],
        "query_ids": ["Q1", "Q2"],
        "queries": [
            "quick fox",
            "passage retrieval",
        ],
        "qrels": {
            "Q1": {"D1": 1},
            "Q2": {"D2": 1},
        },
    }


def _find_file(
    raw_dir: Path,
    candidates: List[str],
) -> Optional[Path]:
    for candidate in candidates:
        candidate_path = raw_dir / candidate
        if candidate_path.exists():
            return candidate_path

    for candidate in candidates:
        for path in raw_dir.rglob(candidate):
            return path

    return None


def _load_pairs(path: Path) -> Tuple[List[str], List[str]]:
    if path.suffix in {".tsv", ".txt"}:
        return _load_tsv_pairs(path)

    if path.suffix in {".jsonl", ".json"}:
        return _load_json_pairs(path)

    raise ValueError(
        f"Unsupported file type for pair loading: {path.suffix}"
    )


def _load_tsv_pairs(path: Path) -> Tuple[List[str], List[str]]:
    ids: List[str] = []
    texts: List[str] = []

    with path.open("r", encoding="utf-8", errors="replace") as file:
        reader = csv.reader(file, delimiter="\t")
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            if len(row) < 2:
                continue
            ids.append(row[0].strip())
            texts.append(" ".join(cell.strip() for cell in row[1:]).strip())

    return ids, texts


def _load_json_pairs(path: Path) -> Tuple[List[str], List[str]]:
    objects = list(_iter_json_objects(path))
    ids: List[str] = []
    texts: List[str] = []

    for obj in objects:
        if "query_id" in obj and "query" in obj:
            ids.append(str(obj["query_id"]))
            texts.append(str(obj["query"]))
            continue
        if "id" in obj and "passage" in obj:
            ids.append(str(obj["id"]))
            texts.append(str(obj["passage"]))
            continue
        if "id" in obj and "text" in obj:
            ids.append(str(obj["id"]))
            texts.append(str(obj["text"]))
            continue
        if "doc_id" in obj and "text" in obj:
            ids.append(str(obj["doc_id"]))
            texts.append(str(obj["text"]))
            continue

        string_keys = [
            (k, v)
            for k, v in obj.items()
            if isinstance(v, str)
        ]
        if len(string_keys) >= 2:
            ids.append(str(string_keys[0][1]))
            texts.append(str(string_keys[1][1]))

    return ids, texts


def _iter_json_objects(path: Path) -> Iterable[Dict[str, object]]:
    with path.open("r", encoding="utf-8", errors="replace") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                try:
                    obj = json.loads(file.read())
                except Exception:
                    continue
            if isinstance(obj, dict):
                yield obj
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, dict):
                        yield item


def _load_qrels(path: Path) -> Dict[str, Dict[str, int]]:
    qrels: Dict[str, Dict[str, int]] = {}

    with path.open("r", encoding="utf-8", errors="replace") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                qid = parts[0]
                doc_id = parts[2] if len(parts) >= 4 else parts[1]
                score = int(parts[-1])
                qrels.setdefault(qid, {})[doc_id] = score
                continue
            try:
                obj = json.loads(line)
                qid = str(obj.get("query_id", obj.get("qid", "")))
                doc_id = str(obj.get("doc_id", obj.get("pid", obj.get("docid", ""))))
                score = int(obj.get("score", obj.get("label", 1)))
                qrels.setdefault(qid, {})[doc_id] = score
            except json.JSONDecodeError:
                continue

    return qrels
