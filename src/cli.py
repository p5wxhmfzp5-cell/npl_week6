"""
CLI Entry Point

CPU-first retrieval benchmarking framework.

Commands
--------
download    Download datasets
index       Build retrieval indexes
run         Run benchmark experiment
analyze     Generate analysis outputs
info        Inspect configuration
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from benchmark.runner import (
    BenchmarkRunner
)

from utils.config import load_yaml_config
from utils.logging import get_logger

console = Console()
logger = get_logger(__name__)

app = typer.Typer(
    add_completion=False,
    help="Retrieval Benchmark Framework CLI",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


############################################################
# DOWNLOAD
############################################################

@app.command()
def download(
    dataset: str = typer.Option(
        ...,
        "--dataset",
        "-d",
        help="Dataset name: msmarco | scifact",
    ),
):
    """
    Download benchmark datasets.
    """

    console.rule("[bold blue]Dataset Download")

    dataset = dataset.lower()

    if dataset == "msmarco":

        from datasets.msmarco import download_msmarco

        download_msmarco()

    elif dataset == "scifact":

        from datasets.scifact import download_scifact

        download_scifact()

    else:
        raise typer.BadParameter(
            f"Unsupported dataset: {dataset}"
        )

    console.print(
        f"[green]✓ Download completed: {dataset}"
    )


############################################################
# INDEX
############################################################

@app.command()
def index(
    method: str = typer.Option(
        ...,
        "--method",
        "-m",
        help="Retrieval method",
    ),
    dataset: str = typer.Option(
        ...,
        "--dataset",
        "-d",
    ),
):
    """
    Build retrieval index.
    """

    console.rule("[bold blue]Index Builder")

    logger.info(
        f"method={method} dataset={dataset}"
    )

    from retrieval.registry import get_retriever
    from datasets.loader import load_dataset

    data = load_dataset(dataset)

    retriever = get_retriever(method)

    retriever.fit(
        doc_ids=data["doc_ids"],
        documents=data["documents"],
    )

    retriever.build_index()

    save_dir = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "indexes"
        / dataset
        / method
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    retriever.save(
        str(save_dir)
    )

    console.print(
        f"[green]✓ Index built: {method}"
    )


############################################################
# RUN EXPERIMENT
############################################################

@app.command()
def run(
    config: str = typer.Option(
        ...,
        "--config",
        "-c",
        help="Experiment YAML path",
    ),
):
    """
    Run benchmark experiment.
    """

    console.rule(
        "[bold blue]Experiment Runner"
    )

    config_path = Path(config)

    cfg = load_yaml_config(
        config_path
    )

    from benchmark.runner import BenchmarkRunner

    runner = BenchmarkRunner(
        config=cfg
    )

    runner.run()

    console.print(
        "[green]✓ Benchmark finished"
    )


############################################################
# ANALYZE
############################################################

@app.command()
def analyze(
    results: str = typer.Option(
        ...,
        "--results",
        "-r",
        help="Metrics CSV file",
    ),
):
    """
    Generate benchmark analysis.
    """

    console.rule(
        "[bold blue]Analysis"
    )

    from analysis.plots import (
        generate_all_plots,
    )

    generate_all_plots(
        results_path=results
    )

    console.print(
        "[green]✓ Analysis completed"
    )


############################################################
# INFO
############################################################

@app.command()
def info(
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
    ),
):
    """
    Display configuration details.
    """

    table = Table(
        title="Framework Info"
    )

    table.add_column(
        "Field",
        style="cyan",
    )

    table.add_column(
        "Value",
        style="green",
    )

    table.add_row(
        "Project Root",
        str(PROJECT_ROOT),
    )

    if config:

        cfg = load_yaml_config(
            config
        )

        table.add_row(
            "Experiment",
            cfg.get(
                "experiment_name",
                "unknown",
            ),
        )

        table.add_row(
            "Dataset",
            cfg.get(
                "dataset",
                "unknown",
            ),
        )

    console.print(table)


############################################################
# ENTRYPOINT
############################################################

def main():

    app()


if __name__ == "__main__":

    main()