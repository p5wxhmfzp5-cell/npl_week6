"""Analysis plot generation utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def generate_all_plots(results_path: str) -> None:
    results_path = Path(results_path)

    if not results_path.exists():
        raise FileNotFoundError(
            f"Results file not found: {results_path}"
        )

    df = pd.read_csv(results_path)

    if "method" not in df.columns:
        raise ValueError(
            "Results CSV must contain a 'method' column for plotting."
        )

    plot_dir = results_path.parent / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        column
        for column in df.columns
        if column != "method"
    ]

    if not metrics:
        raise ValueError(
            "No numeric metrics found in results CSV to plot."
        )

    for metric in metrics:
        if df[metric].dtype == object:
            continue

        fig, ax = plt.subplots(figsize=(8, 4))
        df.plot.bar(
            x="method",
            y=metric,
            ax=ax,
            legend=False,
            color="tab:blue",
        )
        ax.set_title(f"Benchmark Metric: {metric}")
        ax.set_ylabel(metric)
        ax.set_xlabel("Method")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()

        outfile = plot_dir / f"{metric.replace('@', '_').replace('/', '_').replace(' ', '_')}.png"
        fig.savefig(outfile)
        plt.close(fig)

    summary_path = plot_dir / "plot_summary.txt"
    summary_path.write_text(
        "Generated plots for metrics: "
        + ", ".join(metrics)
        + "\n"
        + f"Source results: {results_path.name}\n"
    )
