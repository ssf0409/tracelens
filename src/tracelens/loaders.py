"""CSV and JSONL task loaders for TraceLens.

Two :class:`~tracelens.core.task.TaskLoader` implementations, stdlib-only —
the parsing is delegated to :mod:`json` and :mod:`csv`; these classes own
only the row-to-:class:`~tracelens.core.task.Task` mapping:

- :class:`JSONLTaskLoader` — one JSON object per line (``.jsonl``).
- :class:`CSVTaskLoader` — tabular data via :mod:`csv.DictReader`.

Both accept configurable field/column names so you can point them at your
own files without renaming columns. For hosted datasets (HuggingFace etc.)
see the recipe in ``docs/task-sources.md`` — hosted integrations wrap their
ecosystem's maintained client rather than living in core.

Example::

    from tracelens import CSVTaskLoader, JSONLTaskLoader

    # JSONL — each line must have at least an "input" key
    tasks = JSONLTaskLoader().load("eval_suite.jsonl")

    # CSV — minimal file with an "input" column
    tasks = CSVTaskLoader().load("eval_suite.csv")

    # Custom column names
    tasks = CSVTaskLoader(
        input_col="prompt",
        metadata_cols=["category", "difficulty"],
    ).load("my_data.csv")
"""

import csv
import json
from pathlib import Path
from typing import Any

from tracelens.core.task import Task, TaskLoader


class JSONLTaskLoader(TaskLoader):
    """Load and save tasks from/to newline-delimited JSON (``.jsonl``) files.

    Each line must be a valid JSON object.  By default the loader reads the
    ``"input"`` key as :attr:`~tracelens.core.task.Task.input_data` and wraps
    every other key as :attr:`~tracelens.core.task.Task.metadata`.  Pass
    *metadata_fields* to select only certain keys for metadata (all remaining
    non-input, non-reserved keys are ignored).

    Args:
        input_field: JSON key whose value becomes ``Task.input_data``.
            The value may be a dict (used as-is) or a scalar/list (wrapped as
            ``{"value": <scalar>}``).
        metadata_fields: Optional list of JSON keys to collect into
            ``Task.metadata``.  When *None* (default), **all** keys except
            *input_field* and the Task reserved keys are included.

    Reserved keys that are forwarded directly to :class:`~tracelens.core.task.Task`
    fields rather than being placed in ``metadata``:
    ``task_id``, ``name``, ``description``, ``tags``, ``difficulty``,
    ``category``, ``timeout_seconds``.
    """

    # Task constructor kwargs that should be forwarded verbatim, not stashed in metadata.
    _RESERVED: frozenset[str] = frozenset(
        {
            "task_id",
            "name",
            "description",
            "tags",
            "difficulty",
            "category",
            "timeout_seconds",
            "expectation",
            "metadata",
        }
    )

    def __init__(
        self,
        input_field: str = "input",
        metadata_fields: list[str] | None = None,
    ) -> None:
        self.input_field = input_field
        self.metadata_fields = metadata_fields

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _row_to_task(self, row: dict[str, Any]) -> Task:
        """Convert a parsed JSONL object to a :class:`Task`."""
        raw_input = row.get(self.input_field, {})
        # Coerce scalar / list inputs to a mapping so Task.input_data is always a dict.
        if isinstance(raw_input, dict):
            input_data: dict[str, Any] = raw_input
        else:
            input_data = {"value": raw_input}

        # If an explicit 'metadata' key is present (e.g. from a round-trip save),
        # use it directly; otherwise collect metadata from non-reserved keys.
        _skip_for_meta = self._RESERVED | {self.input_field}
        if "metadata" in row:
            metadata: dict[str, Any] = dict(row["metadata"])
        elif self.metadata_fields is not None:
            metadata = {k: row[k] for k in self.metadata_fields if k in row}
        else:
            metadata = {k: v for k, v in row.items() if k not in _skip_for_meta}

        # Forward reserved Task fields that are present in the row (except 'metadata',
        # which is already handled above).
        task_kwargs: dict[str, Any] = {
            k: row[k] for k in self._RESERVED if k in row and k != "metadata"
        }

        # Fall back to a name derived from input when not provided.
        if "name" not in task_kwargs:
            task_kwargs["name"] = str(raw_input)[:80] if raw_input else "unnamed"

        return Task(input_data=input_data, metadata=metadata, **task_kwargs)

    # ------------------------------------------------------------------
    # TaskLoader interface
    # ------------------------------------------------------------------

    def load(self, source: str | Path) -> list[Task]:
        """Load tasks from a ``.jsonl`` file or directory of ``.jsonl`` files.

        Args:
            source: Path to a ``.jsonl`` file, or a directory whose
                ``**/*.jsonl`` files are loaded in sorted order.

        Returns:
            List of :class:`~tracelens.core.task.Task` objects.

        Raises:
            ValueError: If *source* is neither a file nor a directory.
        """
        path = Path(source)

        if path.is_file():
            tasks: list[Task] = []
            with open(path, encoding="utf-8") as fh:
                for lineno, raw in enumerate(fh, start=1):
                    raw = raw.strip()
                    if not raw:
                        continue  # skip blank lines
                    try:
                        obj = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"{path}:{lineno}: invalid JSON — {exc}"
                        ) from exc
                    if not isinstance(obj, dict):
                        raise ValueError(
                            f"{path}:{lineno}: each line must be a JSON object, "
                            f"got {type(obj).__name__}"
                        )
                    tasks.append(self._row_to_task(obj))
            return tasks

        if path.is_dir():
            result: list[Task] = []
            for file in sorted(path.glob("**/*.jsonl")):
                result.extend(self.load(file))
            return result

        raise ValueError(f"Source is not a file or directory: {source!r}")

    def save(self, tasks: list[Task], destination: str | Path) -> None:
        """Save tasks to a ``.jsonl`` file.

        Each task is serialised as a single JSON object on its own line.
        The object always contains an ``"input"`` key (or the configured
        *input_field*) plus all Task fields.

        Args:
            tasks: Tasks to write.
            destination: Output file path.  Parent directories are created
                automatically.
        """
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as fh:
            for task in tasks:
                record: dict[str, Any] = task.model_dump(mode="json")
                # Promote input_data under the configured field name and remove
                # the internal 'input_data' key so round-trips work cleanly.
                record[self.input_field] = record.pop("input_data")
                fh.write(json.dumps(record, default=str) + "\n")


class CSVTaskLoader(TaskLoader):
    """Load and save tasks from/to CSV files.

    Uses :class:`csv.DictReader` / :class:`csv.DictWriter` under the hood, so
    any dialect supported by the standard library is accepted.

    By default the loader reads the ``"input"`` column as
    :attr:`~tracelens.core.task.Task.input_data` (wrapped in
    ``{"value": <cell>}``) and collects all remaining non-reserved columns
    into :attr:`~tracelens.core.task.Task.metadata`.

    Args:
        input_col: CSV column whose value becomes ``Task.input_data``.
        metadata_cols: Optional list of column names to collect into
            ``Task.metadata``.  When *None* (default), **all** columns except
            *input_col* and the Task reserved column names are included.

    Reserved column names that are forwarded to :class:`~tracelens.core.task.Task`
    fields verbatim:
    ``task_id``, ``name``, ``description``, ``tags``, ``difficulty``,
    ``category``, ``timeout_seconds``.
    """

    _RESERVED: frozenset[str] = frozenset(
        {
            "task_id",
            "name",
            "description",
            "tags",
            "difficulty",
            "category",
            "timeout_seconds",
            "expectation",
        }
    )

    # Columns whose cells are JSON/numeric rather than free text.
    _PARSED_RESERVED: frozenset[str] = frozenset(
        {"tags", "timeout_seconds", "expectation"}
    )

    def __init__(
        self,
        input_col: str = "input",
        metadata_cols: list[str] | None = None,
    ) -> None:
        self.input_col = input_col
        self.metadata_cols = metadata_cols

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _try_parse(value: str) -> Any:
        """Attempt to deserialise *value* as JSON; fall back to the raw string."""
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value

    def _row_to_task(self, row: dict[str, str]) -> Task:
        """Convert a :class:`csv.DictReader` row to a :class:`Task`."""
        raw_input = row.get(self.input_col, "")
        parsed_input = self._try_parse(raw_input)

        # Always produce a dict for input_data.
        if isinstance(parsed_input, dict):
            input_data: dict[str, Any] = parsed_input
        else:
            input_data = {"value": parsed_input}

        # Build metadata from requested or all non-reserved columns.
        if self.metadata_cols is not None:
            metadata: dict[str, Any] = {
                k: self._try_parse(row[k]) for k in self.metadata_cols if k in row
            }
        else:
            skip = self._RESERVED | {self.input_col}
            metadata = {
                k: self._try_parse(v) for k, v in row.items() if k not in skip
            }

        # Forward reserved Task fields from row columns. Free-text columns
        # are taken verbatim — a description that happens to be valid JSON
        # ("true", "123") must stay a string; only structured/numeric
        # fields go through JSON parsing.
        task_kwargs: dict[str, Any] = {}
        for key in self._RESERVED:
            if key in row and row[key]:
                task_kwargs[key] = (
                    self._try_parse(row[key])
                    if key in self._PARSED_RESERVED
                    else row[key]
                )

        # Default name to a truncated view of the input.
        if "name" not in task_kwargs:
            task_kwargs["name"] = raw_input[:80] if raw_input else "unnamed"

        return Task(input_data=input_data, metadata=metadata, **task_kwargs)

    # ------------------------------------------------------------------
    # TaskLoader interface
    # ------------------------------------------------------------------

    def load(self, source: str | Path) -> list[Task]:
        """Load tasks from a ``.csv`` file or directory of ``.csv`` files.

        Args:
            source: Path to a ``.csv`` file, or a directory whose
                ``**/*.csv`` files are loaded in sorted order.

        Returns:
            List of :class:`~tracelens.core.task.Task` objects.

        Raises:
            ValueError: If *source* is neither a file nor a directory.
        """
        path = Path(source)

        if path.is_file():
            tasks: list[Task] = []
            with open(path, newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    tasks.append(self._row_to_task(dict(row)))
            return tasks

        if path.is_dir():
            result: list[Task] = []
            for file in sorted(path.glob("**/*.csv")):
                result.extend(self.load(file))
            return result

        raise ValueError(f"Source is not a file or directory: {source!r}")

    def save(self, tasks: list[Task], destination: str | Path) -> None:
        """Save tasks to a ``.csv`` file.

        Task fields are flattened to strings; complex values (dicts, lists) are
        JSON-serialised.  The ``input_data`` dict is stored under the
        configured *input_col* column name.

        Args:
            tasks: Tasks to write.
            destination: Output file path.  Parent directories are created
                automatically.
        """
        if not tasks:
            path = Path(destination)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")
            return

        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Build header: reserved scalar fields + input_col + metadata keys.
        reserved_cols = [
            "task_id",
            "name",
            "description",
            "difficulty",
            "category",
            "timeout_seconds",
            "tags",
            "expectation",
        ]
        # Collect all metadata keys across all tasks (preserving insertion order).
        meta_keys: list[str] = []
        for task in tasks:
            for k in task.metadata:
                if k not in meta_keys:
                    meta_keys.append(k)

        fieldnames = reserved_cols + [self.input_col] + meta_keys

        def _serialise(value: Any) -> str:
            if isinstance(value, (dict, list)):
                return json.dumps(value, default=str)
            if value is None:
                return ""
            return str(value)

        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for task in tasks:
                record: dict[str, str] = {
                    "task_id": _serialise(task.task_id),
                    "name": _serialise(task.name),
                    "description": _serialise(task.description),
                    "difficulty": _serialise(task.difficulty),
                    "category": _serialise(task.category),
                    "timeout_seconds": _serialise(task.timeout_seconds),
                    "tags": _serialise(task.tags),
                    "expectation": (
                        _serialise(task.expectation.model_dump(exclude_none=True))
                        if task.expectation is not None
                        else ""
                    ),
                    self.input_col: _serialise(task.input_data),
                }
                for k in meta_keys:
                    record[k] = _serialise(task.metadata.get(k))
                writer.writerow(record)
