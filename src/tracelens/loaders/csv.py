"""CSV Task loader and CSV-specific value codecs."""

import csv
import json
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError

from tracelens.core.task import Task, TaskLoader
from tracelens.loaders._records import (
    map_record,
    source_files,
    validate_mapping_fields,
)


@lru_cache
def _task_field_adapter(name: str) -> TypeAdapter[Any]:
    return TypeAdapter(Task.model_fields[name].annotation)


def _decode_json_or_text(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _decode_task_cell(name: str, value: str) -> Any:
    adapter = _task_field_adapter(name)
    if value == '""':
        try:
            return adapter.validate_json(value)
        except (ValidationError, ValueError):
            pass
    try:
        return adapter.validate_python(value, strict=True)
    except ValidationError:
        try:
            return adapter.validate_json(value)
        except (ValidationError, ValueError):
            return value


def _decode_csv_record(row: Mapping[str, str | None], input_field: str) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for name, value in row.items():
        if value is None:
            continue
        if name == input_field:
            record[name] = _decode_json_or_text(value)
        elif name == "metadata":
            if value != "":
                record[name] = _decode_json_or_text(value)
        elif name in Task.model_fields and name not in {"input_data", "metadata"}:
            if value != "":
                record[name] = _decode_task_cell(name, value)
        elif value != "":
            record[name] = _decode_json_or_text(value)
    return record


def _encode_task_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return json.dumps(value) if value == "" else value
    return json.dumps(value, default=str)


class CSVTaskLoader(TaskLoader):
    """Load and save Tasks in CSV, with one canonical metadata column."""

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
        for path in source_files(source, ".csv"):
            with open(path, newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                if reader.fieldnames is None:
                    continue
                if self.input_field not in reader.fieldnames:
                    raise ValueError(f"{path}: missing required input field {self.input_field!r}")
                for line_number, row in enumerate(reader, start=2):
                    try:
                        tasks.append(
                            map_record(
                                _decode_csv_record(row, self.input_field),
                                input_field=self.input_field,
                                metadata_fields=self.metadata_fields,
                            )
                        )
                    except ValueError as error:
                        raise ValueError(f"{path}:{line_number}: {error}") from error
        return tasks

    def save(self, tasks: list[Task], destination: str | Path) -> None:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not tasks:
            path.write_text("", encoding="utf-8")
            return

        task_columns = [
            name for name in Task.model_fields if name not in {"input_data", "metadata"}
        ]
        fieldnames = [*task_columns, self.input_field, "metadata"]
        with open(path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for task in tasks:
                dumped = task.model_dump(mode="json")
                record = {
                    name: _encode_task_cell(dumped[name]) for name in task_columns
                }
                record[self.input_field] = json.dumps(dumped["input_data"], default=str)
                record["metadata"] = json.dumps(dumped["metadata"], default=str)
                writer.writerow(record)
