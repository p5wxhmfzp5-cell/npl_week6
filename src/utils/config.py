"""
Configuration Utilities

Responsibilities
----------------
1. Load YAML configs
2. Validate required fields
3. Merge experiment + model + dataset configs
4. Provide dot-access wrapper
5. Reproducible benchmark configuration handling
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    .parent
)


############################################################
# DotConfig
############################################################

class DotConfig(dict):
    """
    Dictionary with attribute access.

    Example
    -------
    cfg.dataset

    instead of

    cfg["dataset"]
    """

    def __getattr__(self, key):

        try:
            value = self[key]

            if isinstance(value, dict):

                return DotConfig(value)

            return value

        except KeyError as exc:

            raise AttributeError(
                key
            ) from exc

    def __setattr__(self, key, value):

        self[key] = value


############################################################
# YAML LOADING
############################################################

def load_yaml_config(
    path: str | Path,
) -> DotConfig:
    """
    Load YAML configuration.

    Parameters
    ----------
    path:
        YAML file path.

    Returns
    -------
    DotConfig
    """

    path = Path(path)

    if not path.exists():

        raise FileNotFoundError(
            f"Config not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        cfg = yaml.safe_load(f)

    return DotConfig(cfg)


############################################################
# VALIDATION
############################################################

def validate_experiment_config(
    cfg: Dict[str, Any],
) -> None:
    """
    Validate experiment schema.
    """

    required = [

        "experiment_name",
        "dataset",
        "methods",
        "evaluation",
    ]

    missing = [

        field
        for field in required
        if field not in cfg
    ]

    if missing:

        raise ValueError(
            "Missing config fields: "
            f"{missing}"
        )


def validate_model_config(
    cfg: Dict[str, Any],
) -> None:
    """
    Validate retriever config.
    """

    required = [

        "name",
        "retriever_class",
        "type",
    ]

    missing = [

        f
        for f in required
        if f not in cfg
    ]

    if missing:

        raise ValueError(
            f"Invalid model config: {missing}"
        )


def validate_dataset_config(
    cfg: Dict[str, Any],
) -> None:
    """
    Validate dataset schema.
    """

    required = [

        "dataset_name",
        "evaluation",
    ]

    missing = [

        field
        for field in required
        if field not in cfg
    ]

    if missing:

        raise ValueError(
            f"Dataset config invalid: "
            f"{missing}"
        )


############################################################
# CONFIG MERGING
############################################################

def merge_configs(
    experiment_cfg: Dict[str, Any],
    dataset_cfg: Optional[
        Dict[str, Any]
    ] = None,
    model_cfg: Optional[
        Dict[str, Any]
    ] = None,
) -> DotConfig:
    """
    Merge configs into unified object.

    Priority:

        model > dataset > experiment
    """

    merged = {}

    merged.update(
        experiment_cfg
    )

    if dataset_cfg:

        merged["dataset_config"] = (
            dataset_cfg
        )

    if model_cfg:

        merged["model_config"] = (
            model_cfg
        )

    return DotConfig(
        merged
    )


############################################################
# PATH HELPERS
############################################################

def config_dir() -> Path:

    return PROJECT_ROOT / "configs"


def dataset_config_path(
    dataset_name: str,
) -> Path:
    """
    Resolve dataset config.
    """

    dataset_name = dataset_name.strip().lower()

    path = (
        config_dir()
        / "datasets"
        / f"{dataset_name}.yaml"
    )

    if not path.exists() and dataset_name == "msmarco":
        fallback = (
            config_dir()
            / "datasets"
            / "msmacro.yaml"
        )
        if fallback.exists():
            return fallback

    return path


def model_config_path(
    model_name: str,
) -> Path:
    """
    Resolve model config.
    """

    return (
        config_dir()
        / "models"
        / f"{model_name}.yaml"
    )


############################################################
# HIGH LEVEL LOADER
############################################################

def load_experiment_bundle(
    experiment_path: str | Path,
) -> DotConfig:
    """
    Fully resolve benchmark config.

    Loads:

    - experiment config
    - dataset config
    - all model configs

    Returns
    -------
    Unified benchmark bundle.
    """

    exp_cfg = load_yaml_config(
        experiment_path
    )

    validate_experiment_config(
        exp_cfg
    )

    dataset_name = exp_cfg.dataset

    dataset_cfg = load_yaml_config(
        dataset_config_path(
            dataset_name
        )
    )

    validate_dataset_config(
        dataset_cfg
    )

    model_cfgs = {}

    for method in exp_cfg.methods:

        cfg = load_yaml_config(
            model_config_path(
                method
            )
        )

        validate_model_config(
            cfg
        )

        model_cfgs[
            method
        ] = cfg

    bundle = {

        "experiment": exp_cfg,
        "dataset": dataset_cfg,
        "models": model_cfgs,
    }

    return DotConfig(
        bundle
    )


############################################################
# EXPORT
############################################################

__all__ = [

    "DotConfig",

    "load_yaml_config",

    "merge_configs",

    "load_experiment_bundle",

    "validate_experiment_config",

    "validate_dataset_config",

    "validate_model_config",
]