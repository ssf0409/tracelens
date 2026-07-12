"""CSV, JSONL, and HuggingFace task loaders for TraceLens.

Provides three :class:`~tracelens.core.task.TaskLoader` implementations:

- :class:`JSONLTaskLoader` — one JSON object per line (``.jsonl``). Stdlib only.
- :class:`CSVTaskLoader` — tabular data via :mod:`csv`. Stdlib only.
- :class:`HFDatasetLoader` — loads a HuggingFace ``datasets.Dataset`` or
  ``DatasetDict`` split. Requires the optional ``[datasets]`` extra::

      pip install "tracelens[datasets]"

Both CSV/JSONL loaders accept configurable field/column names so you can
point them at your own files without renaming columns.

Example::

    from tracelens.loaders import JSONLTaskLoader, CSVTaskLoader, HFDatasetLoader

    # JSONL — each line must have at least an "input" key
    tasks = JSONLTaskLoader().load("eval_suite.jsonl")

    # CSV — minimal file with an "input" column
    tasks = CSVTaskLoader().load("eval_suite.csv")

    # Custom column names
    tasks = CSVTaskLoader(
        input_col="prompt",
        metadata_cols=["category", "difficulty"],
    ).load("my_data.csv")

    # HuggingFace Hub dataset (requires tracelens[datasets])
    tasks = HFDatasetLoader(
        input_field="question",
        metadata_fields=["subject", "level"],
    ).load("cais/mmlu", split="test")
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
    ``category``, ``timeout_seconds``, ``max_retries``.
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
            "max_retries",
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
    ``category``, ``timeout_seconds``, ``max_retries``.
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
            "max_retries",
            "expectation",
        }
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

        # Forward reserved Task fields from row columns.
        task_kwargs: dict[str, Any] = {}
        for key in self._RESERVED:
            if key in row and row[key]:
                task_kwargs[key] = self._try_parse(row[key])

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
        scalar_reserved = [
            "task_id",
            "name",
            "description",
            "difficulty",
            "category",
            "timeout_seconds",
            "max_retries",
        ]
        # Collect all metadata keys across all tasks (preserving insertion order).
        meta_keys: list[str] = []
        for task in tasks:
            for k in task.metadata:
                if k not in meta_keys:
                    meta_keys.append(k)

        fieldnames = scalar_reserved + [self.input_col] + meta_keys

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
                    "max_retries": _serialise(task.max_retries),
                    self.input_col: _serialise(task.input_data),
                }
                for k in meta_keys:
                    record[k] = _serialise(task.metadata.get(k))
                writer.writerow(record)


# ---------------------------------------------------------------------------
# HFDatasetLoader — requires the [datasets] optional extra
# ---------------------------------------------------------------------------

_DATASETS_MISSING_MSG = (
    "HFDatasetLoader requires the 'datasets' package.\n"
    "Install it with:\n\n"
    "    pip install \"tracelens[datasets]\"\n"
    "    # or: uv pip install \"tracelens[datasets]\"\n"
)


class HFDatasetLoader(TaskLoader):
    """Load tasks from a HuggingFace :class:`datasets.Dataset`.

    Requires the optional ``[datasets]`` extra::

        pip install "tracelens[datasets]"

    The loader can be pointed at a Hub dataset identifier (string), a local
    Arrow/Parquet directory, or a pre-loaded :class:`datasets.Dataset` /
    :class:`datasets.DatasetDict` object — the latter is especially useful
    in unit tests.

    Args:
        input_field: Column whose value becomes ``Task.input_data``.
            A dict value is used as-is; a scalar is wrapped as
            ``{"value": <scalar>}``.
        metadata_fields: Columns to collect into ``Task.metadata``.
            When *None* (default), all columns except *input_field* and
            the Task reserved fields are included.
        name: Dataset configuration/subset name (passed to
            :func:`datasets.load_dataset` as *name*).

    Reserved column names that map directly to Task fields (not metadata):
    ``task_id``, ``name``, ``description``, ``tags``, ``difficulty``,
    ``category``, ``timeout_seconds``, ``max_retries``, ``expectation``,
    ``metadata``.

    Example::

        from tracelens.loaders import HFDatasetLoader

        # Load from the Hub
        tasks = HFDatasetLoader(
            input_field="question",
            metadata_fields=["subject", "level"],
        ).load("cais/mmlu", split="test")

        # Load from a local directory
        tasks = HFDatasetLoader().load("./my_dataset", split="train")

        # Pass a pre-loaded Dataset (no network call)
        import datasets as hf
        ds = hf.Dataset.from_list([
            {"input": {"goal": "Write a haiku"}, "name": "Haiku"},
        ])
        tasks = HFDatasetLoader().load(ds)
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
            "max_retries",
            "expectation",
            "metadata",
        }
    )

    def __init__(
        self,
        input_field: str = "input",
        metadata_fields: list[str] | None = None,
        name: str | None = None,
    ) -> None:
        # Fail loudly at construction time — before any network call — so
        # users see the actionable error message immediately.
        try:
            import datasets as _hf_datasets  # noqa: F401
        except ImportError as exc:
            raise ImportError(_DATASETS_MISSING_MSG) from exc

        self.input_field = input_field
        self.metadata_fields = metadata_fields
        self.name = name

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _row_to_task(self, row: dict[str, Any]) -> Task:
        """Convert a dataset row dict to a :class:`Task`."""
        raw_input = row.get(self.input_field, {})
        if isinstance(raw_input, dict):
            input_data: dict[str, Any] = raw_input
        else:
            input_data = {"value": raw_input}

        # Use explicit metadata key if present (round-trip), otherwise collect.
        _skip = self._RESERVED | {self.input_field}
        if "metadata" in row:
            metadata: dict[str, Any] = dict(row["metadata"])
        elif self.metadata_fields is not None:
            metadata = {k: row[k] for k in self.metadata_fields if k in row}
        else:
            metadata = {k: v for k, v in row.items() if k not in _skip}

        task_kwargs: dict[str, Any] = {
            k: row[k] for k in self._RESERVED if k in row and k != "metadata"
        }
        if "name" not in task_kwargs:
            task_kwargs["name"] = str(raw_input)[:80] if raw_input else "unnamed"

        return Task(input_data=input_data, metadata=metadata, **task_kwargs)

    def _iter_dataset(self, dataset: Any, split: str | None) -> Any:
        """Resolve a DatasetDict to a single split, or return the Dataset as-is."""
        import datasets as hf_datasets

        if isinstance(dataset, hf_datasets.DatasetDict):
            if split is None:
                raise ValueError(
                    "Source is a DatasetDict with multiple splits. "
                    "Pass split= to select one, e.g. split='train'."
                )
            if split not in dataset:
                available = list(dataset.keys())
                raise ValueError(
                    f"Split {split!r} not found. Available: {available}"
                )
            return dataset[split]
        return dataset

    # ------------------------------------------------------------------
    # TaskLoader interface
    # ------------------------------------------------------------------

    def load(  # type: ignore[override]
        self,
        source: "str | Path | Any",
        split: str | None = None,
    ) -> list[Task]:
        """Load tasks from a HuggingFace dataset.

        Args:
            source: One of:

                - A Hub dataset identifier string (e.g. ``"cais/mmlu"``).
                - A local path string / :class:`pathlib.Path` pointing to an
                  Arrow/Parquet dataset directory.
                - A pre-loaded :class:`datasets.Dataset` or
                  :class:`datasets.DatasetDict` object.

            split: Dataset split to use (e.g. ``"train"``, ``"test"``).
                Required when *source* is a DatasetDict or a Hub repo with
                multiple splits.

        Returns:
            List of :class:`~tracelens.core.task.Task` objects.
        """
        import datasets as hf_datasets

        if isinstance(source, (str, Path)):
            dataset = hf_datasets.load_dataset(
                str(source),
                name=self.name,
                split=split,
            )
        else:
            # Pre-loaded Dataset or DatasetDict — used in tests and advanced flows.
            dataset = self._iter_dataset(source, split)

        return [self._row_to_task(dict(row)) for row in dataset]

    def save(self, tasks: list[Task], destination: str | Path) -> None:
        """Save tasks as a HuggingFace Arrow dataset directory.

        Args:
            tasks: Tasks to write.
            destination: Directory path to write the Arrow dataset to.
                Created automatically if it does not exist.
        """
        import datasets as hf_datasets

        records = []
        for task in tasks:
            record: dict[str, Any] = task.model_dump(mode="json")
            record[self.input_field] = record.pop("input_data")
            records.append(record)

        ds = hf_datasets.Dataset.from_list(records)
        ds.save_to_disk(str(destination))
