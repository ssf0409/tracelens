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
        input_field="prompt",
        metadata_fields=["category", "difficulty"],
    ).load("my_data.csv")
"""

import csv
import json
import types
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Union, get_args, get_origin

from tracelens.core.task import Task, TaskLoader

# One source of truth for native Task fields. Loaders map every foreign key
# outside this set to Task.metadata unless metadata_fields selects explicitly.
TASK_FIELDS: frozenset[str] = frozenset(Task.model_fields)
_SPECIAL_FIELDS: frozenset[str] = frozenset({"input_data", "metadata"})
_MAPPED_TASK_FIELDS: frozenset[str] = TASK_FIELDS - _SPECIAL_FIELDS
_MISSING = object()


def _is_text_field(name: str) -> bool:
    """True when the Task field is plain text (``str`` / ``str | None``).

    Only unions are unwrapped — ``get_args`` on a generic like
    ``list[str]`` returns its type parameters, which must not be mistaken
    for union members.
    """
    annotation = Task.model_fields[name].annotation
    if annotation is str:
        return True
    if get_origin(annotation) in (Union, types.UnionType):
        args = [a for a in get_args(annotation) if a is not type(None)]
        return len(args) == 1 and args[0] is str
    return False


# Reserved fields whose CSV cells are JSON/numeric rather than free text.
_PARSED_TASK_FIELDS: frozenset[str] = frozenset(
    name for name in _MAPPED_TASK_FIELDS if not _is_text_field(name)
)


def _identity(value: Any) -> Any:
    return value


def _validate_input_field(input_field: str) -> None:
    if input_field in TASK_FIELDS:
        raise ValueError(
            f"input field {input_field!r} conflicts with a Task field; "
            "choose a foreign field name such as 'input' or 'prompt'"
        )


def source_files(source: str | Path, suffix: str) -> list[Path]:
    """Resolve a file or recursively sorted directory into source files."""
    path = Path(source)
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.glob(f"**/*{suffix}"))
    raise ValueError(f"Source is not a file or directory: {source!r}")


def map_record(
    row: Mapping[str, Any],
    *,
    input_field: str = "input",
    metadata_fields: list[str] | None = None,
    decode_input: Callable[[Any], Any] = _identity,
    decode_metadata: Callable[[Any], Any] = _identity,
) -> Task:
    """Map a foreign row to Task while keeping native fields out of metadata."""
    _validate_input_field(input_field)
    if input_field not in row:
        raise ValueError(f"missing required input field {input_field!r}")

    raw_input = row[input_field]
    decoded_input = decode_input(raw_input)
    input_data = (
        dict(decoded_input) if isinstance(decoded_input, Mapping) else {"value": decoded_input}
    )

    def collect_metadata(keys: list[str]) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        for key in keys:
            if key not in row:
                continue
            value = decode_metadata(row[key])
            if value is not _MISSING:
                metadata[key] = value
        return metadata

    if "metadata" in row:
        decoded_metadata = decode_metadata(row["metadata"])
        if decoded_metadata is _MISSING:
            metadata = {}
        elif isinstance(decoded_metadata, Mapping):
            metadata = dict(decoded_metadata)
        else:
            raise ValueError("metadata field must decode to an object")
    elif metadata_fields is not None:
        metadata = collect_metadata(metadata_fields)
    else:
        metadata = collect_metadata(
            [key for key in row if key not in TASK_FIELDS and key != input_field]
        )

    task_kwargs = {key: row[key] for key in _MAPPED_TASK_FIELDS if key in row}
    if "name" not in task_kwargs:
        task_kwargs["name"] = str(raw_input)[:80] if raw_input else "unnamed"

    return Task(input_data=input_data, metadata=metadata, **task_kwargs)


def _try_parse_json(value: Any) -> Any:
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def _decode_csv_metadata(value: Any) -> Any:
    if value == "":
        return _MISSING
    return _try_parse_json(value)


def _prepare_csv_row(row: Mapping[str, Any]) -> dict[str, Any]:
    prepared: dict[str, Any] = {}
    for key, value in row.items():
        if key in _PARSED_TASK_FIELDS:
            if value in ("", None):
                continue
            prepared[key] = _try_parse_json(value)
        elif key in _MAPPED_TASK_FIELDS:
            if value in ("", None):
                continue
            prepared[key] = "" if _is_text_field(key) and value == '""' else value
        else:
            prepared[key] = value
    return prepared


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
    ``task_id``, ``name``, ``description``, ``expectation``, ``tags``,
    ``difficulty``, ``category``, ``timeout_seconds``.
    """

    def __init__(
        self,
        input_field: str = "input",
        metadata_fields: list[str] | None = None,
    ) -> None:
        _validate_input_field(input_field)
        self.input_field = input_field
        self.metadata_fields = metadata_fields

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
        tasks: list[Task] = []
        for path in source_files(source, ".jsonl"):
            with open(path, encoding="utf-8") as fh:
                for lineno, raw in enumerate(fh, start=1):
                    raw = raw.strip()
                    if not raw:
                        continue  # skip blank lines
                    try:
                        obj = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{path}:{lineno}: invalid JSON — {exc}") from exc
                    if not isinstance(obj, dict):
                        raise ValueError(
                            f"{path}:{lineno}: each line must be a JSON object, "
                            f"got {type(obj).__name__}"
                        )
                    try:
                        tasks.append(
                            map_record(
                                obj,
                                input_field=self.input_field,
                                metadata_fields=self.metadata_fields,
                            )
                        )
                    except ValueError as exc:
                        raise ValueError(f"{path}:{lineno}: {exc}") from exc
        return tasks

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
        input_field: CSV column whose value becomes ``Task.input_data``.
        metadata_fields: Optional list of column names to collect into
            ``Task.metadata``.  When *None* (default), **all** columns except
            *input_field* and the Task reserved column names are included.

    Reserved column names — every :class:`~tracelens.core.task.Task` model
    field except ``input_data``/``metadata``, derived from the model — are
    forwarded to Task fields. Plain-text fields are taken verbatim;
    structured/numeric fields are JSON-parsed.
    """

    def __init__(
        self,
        input_field: str = "input",
        metadata_fields: list[str] | None = None,
    ) -> None:
        _validate_input_field(input_field)
        self.input_field = input_field
        self.metadata_fields = metadata_fields

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
        tasks: list[Task] = []
        for path in source_files(source, ".csv"):
            with open(path, newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                fieldnames = reader.fieldnames
                if fieldnames is None:
                    continue
                if self.input_field not in fieldnames:
                    raise ValueError(f"{path}: missing required input field {self.input_field!r}")
                for lineno, row in enumerate(reader, start=2):
                    prepared = _prepare_csv_row(row)
                    try:
                        tasks.append(
                            map_record(
                                prepared,
                                input_field=self.input_field,
                                metadata_fields=self.metadata_fields,
                                decode_input=_try_parse_json,
                                decode_metadata=_decode_csv_metadata,
                            )
                        )
                    except ValueError as exc:
                        raise ValueError(f"{path}:{lineno}: {exc}") from exc
        return tasks

    def save(self, tasks: list[Task], destination: str | Path) -> None:
        """Save tasks to a ``.csv`` file.

        Task fields are flattened to strings; complex values (dicts, lists) are
        JSON-serialised.  The ``input_data`` dict is stored under the
        configured *input_field* column name.

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

        # Build header: reserved Task fields (model declaration order) +
        # input_field + metadata keys.
        reserved_cols = [name for name in Task.model_fields if name not in _SPECIAL_FIELDS]
        # Collect all metadata keys across all tasks (preserving insertion order).
        meta_keys: list[str] = []
        for task in tasks:
            for k in task.metadata:
                if k in TASK_FIELDS or k == self.input_field:
                    raise ValueError(f"metadata key {k!r} conflicts with a Task or input field")
                if k not in meta_keys:
                    meta_keys.append(k)

        fieldnames = reserved_cols + [self.input_field] + meta_keys

        def _serialise_task_field(name: str, value: Any) -> str:
            if value is None:
                return ""
            if name in _PARSED_TASK_FIELDS:
                return json.dumps(value, default=str)
            if value == "":
                return json.dumps(value)
            return str(value)

        def _serialise_metadata(value: Any) -> str:
            return json.dumps(value, default=str)

        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for task in tasks:
                dumped = task.model_dump(mode="json")
                record: dict[str, str] = {
                    col: _serialise_task_field(col, dumped[col]) for col in reserved_cols
                }
                record[self.input_field] = json.dumps(dumped["input_data"], default=str)
                for k in meta_keys:
                    record[k] = (
                        _serialise_metadata(dumped["metadata"][k])
                        if k in dumped["metadata"]
                        else ""
                    )
                writer.writerow(record)
