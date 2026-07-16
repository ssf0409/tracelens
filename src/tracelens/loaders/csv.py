"""CSV Task loader and CSV-specific value codecs."""

import csv
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

from pydantic import TypeAdapter, ValidationError

from tracelens._paths import prepare_destination_path, source_files
from tracelens.core.task import Task, TaskLoader
from tracelens.loaders._records import (
    map_record,
    task_to_record,
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


def _validate_csv_header(path: Path, fieldnames: Sequence[str], input_field: str) -> None:
    blank_columns = [name for name in fieldnames if not name.strip()]
    if blank_columns:
        raise ValueError(f"{path}: blank CSV column names are not allowed")

    duplicate_columns = [
        name for name, count in Counter(fieldnames).items() if count > 1
    ]
    if duplicate_columns:
        names = ", ".join(repr(name) for name in duplicate_columns)
        raise ValueError(f"{path}: duplicate CSV column names: {names}")

    if input_field not in fieldnames:
        raise ValueError(f"{path}: missing required input field {input_field!r}")


def _decode_csv_record(
    row: Mapping[str | None, str | list[str] | None], input_field: str
) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for name, value in row.items():
        if name is None:
            raise ValueError("row has more values than the CSV header")
        if value is None:
            continue
        # DictReader emits list values only for overflow under the None key above.
        cell = cast(str, value)
        if name == input_field:
            record[name] = _decode_json_or_text(cell)
        elif name == "metadata":
            if cell != "":
                record[name] = _decode_json_or_text(cell)
        elif name in Task.model_fields and name not in {"input_data", "metadata"}:
            if cell != "":
                record[name] = _decode_task_cell(name, cell)
        elif cell != "":
            record[name] = _decode_json_or_text(cell)
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
                _validate_csv_header(path, reader.fieldnames, self.input_field)
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
        path = prepare_destination_path(destination)
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
                record = task_to_record(task, input_field=self.input_field)
                encoded_record = {name: _encode_task_cell(record[name]) for name in fieldnames}
                writer.writerow(encoded_record)
