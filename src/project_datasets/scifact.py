"""SciFact dataset utilities."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from utils.config import (
    dataset_config_path,
    load_yaml_config,
)


def download_scifact() -> None:
    raise NotImplementedError(
        "SciFact download is not implemented in this repository skeleton. "
        "Add a downloader or populate data/raw/scifact manually."
    )


def load_scifact() -> Dict[str, list]:
    cfg = load_yaml_config(
        dataset_config_path("scifact")
    )

    raw_dir = Path(cfg.storage.raw_dir)

    if not raw_dir.exists() or not raw_dir.is_dir():
        return _sample_scifact()

    query_file = _find_file(
        raw_dir,
        [
            "queries.tsv",
            "queries.jsonl",
            "claims.jsonl",
            "scifact.jsonl",
            "scifact.json",
        ],
    )
    corpus_file = _find_file(
        raw_dir,
        [
            "corpus.tsv",
            "docs.tsv",
            "abstracts.tsv",
            "corpus.jsonl",
            "abstracts.jsonl",
        ],
    )
    qrels_file = _find_file(
        raw_dir,
        [
            "qrels.tsv",
            "qrels.jsonl",
            "qrels.dev.tsv",
            "qrels.dev.small.tsv",
        ],
    )

    if query_file and corpus_file and qrels_file:
        query_ids, queries = _load_pairs(query_file)
        doc_ids, documents = _load_pairs(corpus_file)
        qrels = _load_qrels(qrels_file)
        return {
            "doc_ids": doc_ids,
            "documents": documents,
            "query_ids": query_ids,
            "queries": queries,
            "qrels": qrels,
        }

    json_data_file = _find_file(raw_dir, ["scifact.jsonl", "scifact.json"])
    if json_data_file is not None:
        return _load_scifact_json(json_data_file)

    return _sample_scifact()


def _sample_scifact() -> Dict[str, list]:
    return {
        "doc_ids": ["D1", "D2", "D3"],
        "documents": [
            "The hypothesis is tested against scientific evidence.",
            "SciFact contains scientific claims and supporting abstracts.",
            "This sample document demonstrates local dataset loading.",
        ],
        "query_ids": ["Q1", "Q2"],
        "queries": [
            "scientific hypothesis",
            "claim verification",
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
        if "id" in obj and "claim" in obj:
            ids.append(str(obj["id"]))
            texts.append(str(obj["claim"]))
            continue
        if "id" in obj and "text" in obj:
            ids.append(str(obj["id"]))
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


def _load_scifact_json(path: Path) -> Dict[str, list]:
    objects = list(_iter_json_objects(path))
    doc_texts: Dict[str, str] = {}
    query_ids: List[str] = []
    queries: List[str] = []
    qrels: Dict[str, Dict[str, int]] = {}

    for obj in objects:
        if "claim" in obj:
            qid = str(obj.get("id", obj.get("query_id", len(query_ids) + 1)))
            query = str(obj["claim"])
            query_ids.append(qid)
            queries.append(query)

            evidences = obj.get("evidence", obj.get("evidences", []))
            if isinstance(evidences, dict):
                evidence_items = evidences.items()
            elif isinstance(evidences, list):
                evidence_items = [
                    (str(item.get("doc_id", item.get("id", ""))), 1)
                    for item in evidences
                    if isinstance(item, dict)
                ]
            else:
                evidence_items = []

            for doc_id, score in evidence_items:
                if doc_id:
                    qrels.setdefault(qid, {})[doc_id] = int(score)

        if "abstracts" in obj and isinstance(obj["abstracts"], list):
            for abstract in obj["abstracts"]:
                if isinstance(abstract, dict) and "id" in abstract and "text" in abstract:
                    doc_texts[str(abstract["id"])] = str(abstract["text"])
        if "corpus" in obj and isinstance(obj["corpus"], list):
            for document in obj["corpus"]:
                if isinstance(document, dict) and "id" in document and "text" in document:
                    doc_texts[str(document["id"])] = str(document["text"])

    if not query_ids or not doc_texts:
        return _sample_scifact()

    doc_ids = list(doc_texts.keys())
    documents = [doc_texts[doc_id] for doc_id in doc_ids]

    if not qrels:
        for qid in query_ids:
            qrels[qid] = {doc_ids[0]: 1}

    return {
        "doc_ids": doc_ids,
        "documents": documents,
        "query_ids": query_ids,
        "queries": queries,
        "qrels": qrels,
    }
