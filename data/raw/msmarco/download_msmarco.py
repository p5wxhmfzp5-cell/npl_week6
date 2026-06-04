from pathlib import Path
from datasets import load_dataset

def download_msmarco():
    ds = load_dataset(
        "ms_marco",
        "v1.1",
        split="validation"
    )

    raw_dir = Path("data/raw/msmarco")
    raw_dir.mkdir(parents=True, exist_ok=True)

    # write queries.tsv
    # write corpus.tsv
    # write qrels.tsv