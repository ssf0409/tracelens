"""Load a local eval set by file format.

``tracelens run --eval-set PATH`` accepts the three local formats TraceLens
ships loaders for. The format is inferred from the file suffix; a directory
needs it spelled out because a folder can hold several formats, and TraceLens
never guesses. Hugging Face Hub datasets stay a Python-API concern
(``HFDatasetLoader``): a Hub ID is not a local path and is not dispatched here.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from tracelens.core.task import JSONTaskLoader, Task, TaskLoader
from tracelens.loaders.csv import CSVTaskLoader
from tracelens.loaders.jsonl import JSONLTaskLoader

EvalSetFormat = Literal["json", "jsonl", "csv"]
EVAL_SET_FORMATS: tuple[EvalSetFormat, ...] = ("json", "jsonl", "csv")
_SUFFIXES: dict[str, EvalSetFormat] = {".json": "json", ".jsonl": "jsonl", ".csv": "csv"}


class EvalSetLoadError(ValueError):
    """A local eval set could not be loaded. The message is user-facing."""


def detect_format(path: Path) -> EvalSetFormat:
    """Infer the eval-set format from a file suffix.

    Raises:
        EvalSetLoadError: For a directory (the format must be given) or an
            unsupported suffix.
    """
    if path.is_dir():
        raise EvalSetLoadError(
            f"{path} is a directory; pass --eval-set-format json|jsonl|csv to say "
            "which files inside it to load"
        )
    suffix = path.suffix.lower()
    if suffix not in _SUFFIXES:
        raise EvalSetLoadError(
            f"unsupported eval-set file type {suffix or '(none)'!r} for {path}; use a "
            ".json, .jsonl, or .csv file, or pass --eval-set-format"
        )
    return _SUFFIXES[suffix]


def make_loader(
    fmt: EvalSetFormat,
    *,
    input_field: str = "input",
    metadata_fields: Sequence[str] | None = None,
) -> TaskLoader:
    """Build the loader for ``fmt`` with the given foreign-record options."""
    if fmt == "json":
        if input_field != "input" or metadata_fields:
            raise EvalSetLoadError(
                "--input-field and --metadata-fields apply to jsonl and csv eval "
                "sets only; JSON eval sets use the native Task shape"
            )
        return JSONTaskLoader()
    if fmt == "jsonl":
        return JSONLTaskLoader(input_field=input_field, metadata_fields=metadata_fields)
    if fmt == "csv":
        return CSVTaskLoader(input_field=input_field, metadata_fields=metadata_fields)
    raise EvalSetLoadError(f"unknown eval-set format {fmt!r}")


def load_tasks(
    source: str | Path,
    *,
    format: EvalSetFormat | None = None,
    input_field: str = "input",
    metadata_fields: Sequence[str] | None = None,
) -> list[Task]:
    """Load tasks from a local ``.json``, ``.jsonl``, or ``.csv`` path.

    Args:
        source: A file, or a directory when ``format`` is given (every file
            with that format's suffix inside it is loaded, recursively).
        format: Explicit format; inferred from the file suffix when omitted.
        input_field: jsonl/csv only: the column holding the task input.
        metadata_fields: jsonl/csv only: foreign columns kept in
            ``Task.metadata``; all of them by default.

    Raises:
        EvalSetLoadError: Missing path, unsupported suffix, directory without
            a format, options that do not apply to the format, invalid JSON,
            or an invalid record. Messages name the file and, where the
            loader knows it, the line.
    """
    path = Path(source)
    if not path.exists():
        raise EvalSetLoadError(f"eval-set path not found: {source}")
    fmt = format if format is not None else detect_format(path)
    try:
        loader = make_loader(fmt, input_field=input_field, metadata_fields=metadata_fields)
    except ValueError as exc:  # loader-level option validation
        raise EvalSetLoadError(str(exc)) from exc
    try:
        return loader.load(path)
    except json.JSONDecodeError as exc:
        raise EvalSetLoadError(f"invalid JSON in eval-set file {source}: {exc}") from exc
    except ValidationError as exc:
        raise EvalSetLoadError(f"invalid task record in {source}: {exc}") from exc
    except ValueError as exc:
        raise EvalSetLoadError(f"could not load eval set {source}: {exc}") from exc
