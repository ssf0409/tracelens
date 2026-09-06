"""Flat Task loaders for local files and optional external sources."""

from tracelens.loaders.csv import CSVTaskLoader
from tracelens.loaders.dispatch import (
    EVAL_SET_FORMATS,
    EvalSetFormat,
    EvalSetLoadError,
    load_tasks,
)
from tracelens.loaders.hf import HFDatasetLoader
from tracelens.loaders.jsonl import JSONLTaskLoader

__all__ = [
    "CSVTaskLoader",
    "EVAL_SET_FORMATS",
    "EvalSetFormat",
    "EvalSetLoadError",
    "HFDatasetLoader",
    "JSONLTaskLoader",
    "load_tasks",
]
