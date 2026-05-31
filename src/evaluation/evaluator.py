"""
Benchmark Evaluator

Unified evaluation engine for:

- MS MARCO
- SciFact
- Sparse retrieval
- Dense retrieval
- Hybrid retrieval
- Re-ranking pipelines

Metrics:
    MRR@10
    NDCG@10
    Recall@10
    MAP
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pytrec_eval

from utils.logging import get_logger

logger = get_logger(__name__)


class Evaluator:
    """
    Unified benchmark evaluator.
    """

    def __init__(
        self,
        dataset_name: str,
    ):

        self.dataset_name = (
            dataset_name.lower()
        )

        ####################################################
        # metric selection
        ####################################################

        if self.dataset_name == "msmarco":

            self.metric_set = {

                "recip_rank",
                "recall_10",
                "map",
            }

        elif self.dataset_name == "scifact":

            self.metric_set = {

                "ndcg_cut_10",
                "recall_10",
                "map",
            }

        else:

            self.metric_set = {

                "recip_rank",
                "ndcg_cut_10",
                "recall_10",
                "map",
            }

    ########################################################
    # FORMATTERS
    ########################################################

    @staticmethod
    def format_rankings(
        rankings: Dict[
            str,
            List[
                Tuple[str, float]
            ]
        ]
    ) -> Dict:
        """
        Convert benchmark rankings
        -> pytrec_eval format.

        Input
        -----

        {
            qid:[
                (doc_id,score),
                ...
            ]
        }

        Output
        ------

        {
            qid:{
                doc_id:score
            }
        }
        """

        formatted = {}

        for qid, docs in (
            rankings.items()
        ):

            formatted[qid] = {

                doc_id: float(score)

                for doc_id, score
                in docs
            }

        return formatted

    ########################################################
    # PER QUERY
    ########################################################

    def evaluate_per_query(
        self,
        rankings: Dict,
        qrels: Dict,
    ) -> Dict:
        """
        Per-query evaluation.
        """

        formatted = (
            self.format_rankings(
                rankings
            )
        )

        evaluator = (
            pytrec_eval.RelevanceEvaluator(
                qrels,
                self.metric_set,
            )
        )

        return evaluator.evaluate(
            formatted
        )

    ########################################################
    # AGGREGATE
    ########################################################

    def evaluate(
        self,
        rankings: Dict,
        qrels: Dict,
    ) -> Dict:
        """
        Aggregate benchmark metrics.
        """

        scores = (
            self.evaluate_per_query(
                rankings,
                qrels,
            )
        )

        if not scores:

            raise RuntimeError(
                "No evaluation output."
            )

        ####################################################
        # aggregation
        ####################################################

        metrics = {}

        n_queries = len(
            scores
        )

        for qid, vals in (
            scores.items()
        ):

            for metric, value in (
                vals.items()
            ):

                metrics.setdefault(
                    metric,
                    []
                ).append(value)

        aggregated = {

            metric: round(
                sum(values)
                / n_queries,
                4,
            )

            for metric, values
            in metrics.items()
        }

        ####################################################
        # normalize names
        ####################################################

        renamed = {}

        metric_map = {

            "recip_rank":
                "mrr@10",

            "ndcg_cut_10":
                "ndcg@10",

            "recall_10":
                "recall@10",

            "map":
                "map",
        }

        for metric, value in (
            aggregated.items()
        ):

            renamed[
                metric_map.get(
                    metric,
                    metric,
                )
            ] = value

        logger.info(
            "Evaluation complete."
        )

        return renamed

    ########################################################
    # LEADERBOARD TABLE
    ########################################################

    @staticmethod
    def summarize(
        results: List[Dict]
    ) -> List[Dict]:
        """
        Sort benchmark outputs.

        Useful for notebook
        and reporting.
        """

        def primary_score(
            row,
        ):

            return max(

                row.get(
                    "mrr@10",
                    0,
                ),

                row.get(
                    "ndcg@10",
                    0,
                ),
            )

        return sorted(

            results,

            key=primary_score,

            reverse=True,
        )