"""JSONL Task loader."""

import json
from collections.abc import Sequence
from pathlib import Path

from tracelens._paths import prepare_destination_path, source_files
from tracelens.core.task import Task, TaskLoader
from tracelens.loaders._records import (
    map_record,
    task_to_record,
    validate_mapping_fields,
)


class JSONLTaskLoader(TaskLoader):
    """Load and save Tasks as one JSON object per line."""

    def __init__(
        self,
        input_field: str = "input",
        metadata_fields: Sequence[str] | None = None,
    ) -> None:
        validate_mapping_fields(input_field, metadata_fields)
        self.input_field = input_field
        self.metadata_fields = metadata_fields

    def load(self, source: str | Path) -> list[Task]:
        tasks: list[Task] = []
        for path in source_files(source, ".jsonl"):
            with open(path, encoding="utf-8") as file:
                for line_number, line in enumerate(file, start=1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(f"{path}:{line_number}: invalid JSON - {error}") from error
                    if not isinstance(record, dict):
                        raise ValueError(
                            f"{path}:{line_number}: each line must be a JSON object, "
                            f"got {type(record).__name__}"
                        )
                    try:
                        tasks.append(
                            map_record(
                                record,
                                input_field=self.input_field,
                                metadata_fields=self.metadata_fields,
                            )
                        )
                    except ValueError as error:
                        raise ValueError(f"{path}:{line_number}: {error}") from error
        return tasks

    def save(self, tasks: list[Task], destination: str | Path) -> None:
        path = prepare_destination_path(destination)
        with open(path, "w", encoding="utf-8") as file:
            for task in tasks:
                record = task_to_record(task, input_field=self.input_field)
                file.write(json.dumps(record, default=str) + "\n")
