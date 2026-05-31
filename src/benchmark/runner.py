"""
Benchmark Runner

Central experiment orchestration layer.

Responsibilities
----------------
- load experiment bundle
- load dataset
- instantiate retrievers
- build/load indexes
- run retrieval
- compute metrics
- save outputs
- benchmark latency

Used by:

    cli.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from datasets.loader import load_dataset
from evaluation.evaluator import Evaluator
from retrieval.registry import get_retriever
from utils.config import (
    DotConfig,
    load_experiment_bundle,
)
from utils.logging import (
    banner,
    get_experiment_logger,
    log_metrics,
)

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    .parent
)


class BenchmarkRunner:
    """
    Main benchmark engine.
    """

    def __init__(
        self,
        config: str | Dict | DotConfig,
    ):

        ####################################################
        # config loading
        ####################################################

        if isinstance(
            config,
            (str, Path),
        ):

            self.bundle = (
                load_experiment_bundle(
                    config
                )
            )

        else:

            self.bundle = config

        self.exp_cfg = (
            self.bundle.experiment
        )

        self.dataset_cfg = (
            self.bundle.dataset
        )

        self.model_cfgs = (
            self.bundle.models
        )

        ####################################################
        # logger
        ####################################################

        self.logger = (
            get_experiment_logger(
                self.exp_cfg
                .experiment_name
            )
        )

        banner(
            self.logger,
            self.exp_cfg
            .experiment_name,
        )

        ####################################################
        # output dirs
        ####################################################

        self.metrics_dir = Path(
            self.exp_cfg.output
            .metrics_dir
        )

        self.rankings_dir = Path(
            self.exp_cfg.output
            .rankings_dir
        )

        self.metrics_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.rankings_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    ########################################################
    # DATA
    ########################################################

    def load_data(
        self,
    ) -> Dict:
        """
        Load dataset adapter.
        """

        self.logger.info(
            "Loading dataset: %s",
            self.exp_cfg.dataset,
        )

        data = load_dataset(
            self.exp_cfg.dataset
        )

        self.logger.info(
            "queries=%d docs=%d",
            len(data["queries"]),
            len(data["documents"]),
        )

        return data

    ########################################################
    # RETRIEVAL
    ########################################################

    def run_method(
        self,
        method_name: str,
        data: Dict,
    ) -> Dict:
        """
        Execute one retrieval method.
        """

        self.logger.info(
            "Running method=%s",
            method_name,
        )

        retriever = (
            get_retriever(
                method_name
            )
        )

        ####################################################
        # fit/index
        ####################################################

        retriever.fit(

            doc_ids=data[
                "doc_ids"
            ],

            documents=data[
                "documents"
            ],
        )

        retriever.build_index()

        ####################################################
        # retrieval loop
        ####################################################

        rankings = {}

        start = time.perf_counter()

        for qid, query in tqdm(

            zip(
                data["query_ids"],
                data["queries"],
            ),

            total=len(
                data["queries"]
            ),

            desc=method_name,
        ):

            results = (
                retriever.retrieve(
                    query=query,
                    top_k=10,
                )
            )

            rankings[
                qid
            ] = results

        elapsed = (
            time.perf_counter()
            - start
        )

        ####################################################
        # latency profiling
        ####################################################

        latency_stats = (
            retriever
            .benchmark_latency(
                data[
                    "queries"
                ][:50]
            )
        )

        ####################################################
        # evaluate
        ####################################################

        evaluator = Evaluator(
            dataset_name=(
                self.exp_cfg
                .dataset
            )
        )

        metrics = evaluator.evaluate(

            rankings=rankings,

            qrels=data[
                "qrels"
            ],
        )

        metrics.update(
            latency_stats
        )

        metrics[
            "runtime_seconds"
        ] = round(
            elapsed,
            3,
        )

        ####################################################
        # save rankings
        ####################################################

        self.save_rankings(
            method_name,
            rankings,
        )

        log_metrics(
            self.logger,
            metrics,
        )

        return metrics

    ########################################################
    # SAVE
    ########################################################

    def save_rankings(
        self,
        method_name: str,
        rankings: Dict,
    ):
        """
        Persist ranking outputs.
        """

        path = (

            self.rankings_dir
            / f"{method_name}.json"
        )

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                rankings,
                f,
                indent=2,
            )

    def save_metrics(
        self,
        results: List[Dict],
    ):
        """
        Save benchmark metrics.
        """

        df = pd.DataFrame(
            results
        )

        out_file = (
            self.metrics_dir
            / "results.csv"
        )

        df.to_csv(
            out_file,
            index=False,
        )

        self.logger.info(
            "Saved metrics: %s",
            out_file,
        )

    ########################################################
    # MAIN LOOP
    ########################################################

    def run(
        self,
    ):
        """
        Run experiment suite.
        """

        data = self.load_data()

        all_results = []

        for method in (
            self.exp_cfg.methods
        ):

            metrics = (
                self.run_method(
                    method,
                    data,
                )
            )

            metrics[
                "method"
            ] = method

            all_results.append(
                metrics
            )

        self.save_metrics(
            all_results
        )

        self.logger.info(
            "Benchmark completed."
        )