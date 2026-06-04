# Retrieval Benchmark Framework (CPU Edition)

Research-grade retrieval benchmarking framework for analyzing **why retrieval methods win under different metrics and datasets**.

Benchmarked methods:

1. BM25
2. TF-IDF
3. Dense General (`all-MiniLM-L6-v2`)
4. Dense IR-tuned (`msmarco-distilbert-base-v3`)
5. Hybrid Retrieval (BM25 + Dense via RRF)
6. ColBERTv2 (CPU-compatible mode)
7. Cross-Encoder Re-Ranking
8. HyDE (FLAN-T5 + Dense encoder)

Datasets:

- MS MARCO v1.1 dev-small
- BEIR SciFact (300 queries / 5k docs)

Primary metrics:

- MRR@10
- NDCG@10

Secondary metrics:

- Recall@k
- MAP
- Latency
- Throughput
- Index Size
- CPU Memory

---

## Research Goal

This benchmark is designed to answer:

**Why do retrieval approaches win on different datasets and metrics?**

We analyze:

- lexical matching strength
- semantic generalization
- zero-shot transfer
- precision / recall tradeoffs
- late interaction behavior
- reranking gains
- efficiency vs effectiveness

---

## Project Structure

```txt
retrieval-benchmark/
│
├── README.md
├── requirements.txt
├── pyproject.toml
│
├── configs/
│   ├── datasets/
│   │   ├── msmarco.yaml
│   │   └── scifact.yaml
│   │
│   ├── models/
│   │   ├── bm25.yaml
│   │   ├── dense.yaml
│   │   ├── colbert.yaml
│   │   └── hyde.yaml
│   │
│   └── experiments/
│       └── full_benchmark.yaml
│
├── data/
│   ├── raw/
│   │   ├── msmarco/  # expected files: corpus.tsv, queries.tsv, qrels*.tsv
│   │   │   ├── corpus.tsv
│   │   │   ├── queries.tsv
│   │   │   └── qrels.dev.small.tsv
│   │   └── scifact/  # expected files: queries.tsv, corpus.tsv, qrels.tsv or scifact.jsonl
│   │       ├── queries.tsv
│   │       ├── corpus.tsv
│   │       └── qrels.tsv
│   ├── processed/
│   └── cache/
│
├── src/
│   │
│   ├── cli.py
│   │
│   ├── retrieval/
│   │   ├── base.py
│   │   ├── bm25.py
│   │   ├── tfidf.py
│   │   ├── dense.py
│   │   ├── hybrid_rrf.py
│   │   ├── colbert_cpu.py
│   │   ├── rerank.py
│   │   └── hyde.py
│   │
│   ├── datasets/
│   │   ├── msmarco.py
│   │   └── scifact.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── evaluator.py
│   │
│   ├── analysis/
│   │   ├── profiler.py
│   │   └── plots.py
│   │
│   └── utils/
│       ├── config.py
│       ├── cache.py
│       └── logging.py
│
├── notebooks/
│   ├── benchmark_analysis.ipynb
│   └── failure_analysis.ipynb
│
└── reports/
    └── paper_template.md
```

---

## Installation

### Create Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install the project package and CLI entry points:

```bash
pip install -e .
```

For evaluation metrics support, install the optional extra:

```bash
pip install -e .[evaluation]
```

After installation, you can run the CLI with either:

```bash
python src/cli.py --help
retrieval-bench --help
retrieval-benchmark --help
```

---

## Quickstart

### 1. Download datasets

MS MARCO:

```bash
python src/cli.py download \
    --dataset msmarco
```

SciFact:

```bash
python src/cli.py download \
    --dataset scifact
```

---

## Data Preparation

Place raw dataset files under `data/raw/<dataset>/` using the naming conventions below.

- `data/raw/msmarco/`
  - `corpus.tsv`
  - `queries.tsv`
  - `qrels*.tsv`
- `data/raw/scifact/`
  - `queries.tsv`
  - `corpus.tsv`
  - `qrels.tsv` or `scifact.jsonl`

If no raw files exist, the loader falls back to a small sample dataset for local testing.

---

### 2. Build indexes

BM25:

```bash
python src/cli.py index \
    --method bm25 \
    --dataset msmarco
```

Dense:

```bash
python src/cli.py index \
    --method dense_m3 \
    --dataset msmarco
```

---

### 3. Run single experiment

```bash
python src/cli.py run \
    --config configs/experiments/full_benchmark.yaml
```

---

### 4. Generate analysis report

```bash
python src/cli.py analyze \
    --results outputs/results.csv
```

---

## Supported Retrieval Methods

| Method | Type | CPU Status |
|--------|------|------|
| BM25 | Sparse | ✓ |
| TF-IDF | Sparse | ✓ |
| MiniLM | Dense | ✓ |
| M3 | Dense | ✓ |
| Hybrid RRF | Hybrid | ✓ |
| ColBERTv2 | Late Interaction | ✓ (reduced mode) |
| Cross Encoder | Reranker | ✓ |
| HyDE | Query Expansion | ✓ |

---

## Expected Findings

### MS MARCO

Expected winners:

- BM25 strong MRR@10
- Cross-Encoder highest precision
- Hybrid robust overall

Reason:

MS MARCO contains strong lexical overlap and single-answer retrieval.

---

### SciFact

Expected winners:

- Dense retrieval
- HyDE
- ColBERT

Reason:

Scientific terminology and semantic variation reward semantic retrieval.

---

## Reproducibility

Fixed seeds:

```python
SEED=42
```

Deterministic settings:

```python
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
```

---

## Citation

If using this framework:

```bibtex
@software{retrieval_benchmark_cpu,
  title={Retrieval Benchmark Framework},
  year={2026}
}
```
