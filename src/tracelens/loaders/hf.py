"""Optional Hugging Face datasets Task loader."""

from collections.abc import Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any

from tracelens._paths import prepare_destination_path
from tracelens.core.task import Task, TaskLoader
from tracelens.loaders._records import (
    map_record,
    task_to_record,
    validate_mapping_fields,
)

_DATASETS_MISSING_MESSAGE = (
    "HFDatasetLoader requires the optional 'datasets' dependency. "
    'Install it with: pip install "tracelens[datasets]"'
)


def _require_datasets() -> Any:
    """Import the optional dependency without affecting core package imports."""
    try:
        return import_module("datasets")
    except ModuleNotFoundError as error:
        if error.name != "datasets":
            raise
        raise ImportError(_DATASETS_MISSING_MESSAGE) from error


def _select_split(dataset: Any, split: str | None, dataset_dict_type: type[Any]) -> Any:
    """Resolve a local DatasetDict while leaving a Dataset unchanged."""
    if not isinstance(dataset, dataset_dict_type):
        return dataset

    available = list(dataset.keys())
    if split is None:
        raise ValueError(f"split is required for DatasetDict; available splits: {available}")
    if split not in dataset:
        raise ValueError(f"split {split!r} not found; available splits: {available}")
    return dataset[split]


class HFDatasetLoader(TaskLoader):
    """Load Hub or locally saved Hugging Face datasets as Tasks.

    String sources are treated as Hub dataset identifiers and require an
    explicit split. Path sources are loaded with ``datasets.load_from_disk``.
    Streaming and Hub writes are intentionally outside this loader's contract.
    """

    def __init__(
        self,
        input_field: str = "input",
        metadata_fields: Sequence[str] | None = None,
        *,
        config_name: str | None = None,
        split: str | None = None,
        revision: str | None = None,
    ) -> None:
        validate_mapping_fields(input_field, metadata_fields)
        self.input_field = input_field
        self.metadata_fields = metadata_fields
        self.config_name = config_name
        self.split = split
        self.revision = revision

    def load(self, source: str | Path) -> list[Task]:
        if isinstance(source, str) and self.split is None:
            raise ValueError("split is required for a Hugging Face Hub dataset")

        datasets = _require_datasets()
        if isinstance(source, Path):
            dataset = datasets.load_from_disk(str(source))
            dataset = _select_split(dataset, self.split, datasets.DatasetDict)
        else:
            dataset = datasets.load_dataset(
                source,
                name=self.config_name,
                split=self.split,
                revision=self.revision,
            )

        source_context = str(source)
        if self.split is not None:
            source_context = f"{source_context}:{self.split}"

        tasks: list[Task] = []
        for row_number, row in enumerate(dataset, start=1):
            try:
                if not isinstance(row, Mapping):
                    raise ValueError("row must be a mapping")
                tasks.append(
                    map_record(
                        row,
                        input_field=self.input_field,
                        metadata_fields=self.metadata_fields,
                    )
                )
            except ValueError as error:
                raise ValueError(f"{source_context} row {row_number}: {error}") from error
        return tasks

    def save(self, tasks: list[Task], destination: str | Path) -> None:
        if not tasks:
            raise ValueError("HFDatasetLoader cannot save an empty Task list")
        datasets = _require_datasets()
        path = prepare_destination_path(destination)
        records = [task_to_record(task, input_field=self.input_field) for task in tasks]
        dataset = datasets.Dataset.from_list(records)
        dataset.save_to_disk(str(path))
