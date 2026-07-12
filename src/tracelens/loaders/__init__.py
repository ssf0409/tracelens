"""Flat Task loaders for local JSONL and CSV sources."""

from tracelens.loaders._records import map_record, source_files
from tracelens.loaders.csv import CSVTaskLoader
from tracelens.loaders.jsonl import JSONLTaskLoader

__all__ = ["CSVTaskLoader", "JSONLTaskLoader", "map_record", "source_files"]
