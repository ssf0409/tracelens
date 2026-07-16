"""Shared mapping between foreign records and Tasks."""

from collections.abc import Mapping, Sequence
from typing import Any

from tracelens.core.task import Task


def validate_mapping_fields(
    input_field: str, metadata_fields: Sequence[str] | None
) -> None:
    """Reject ambiguous input and metadata field configuration."""
    if input_field in Task.model_fields:
        raise ValueError(
            f"input field {input_field!r} conflicts with a Task field; "
            "choose a foreign field name such as 'input' or 'prompt'"
        )
    if metadata_fields is None:
        return
    for field in metadata_fields:
        if field == input_field:
            raise ValueError("metadata_fields cannot include the input field")
        if field in Task.model_fields:
            raise ValueError("metadata_fields cannot include a Task field")


def task_to_record(task: Task, *, input_field: str = "input") -> dict[str, Any]:
    """Serialize a Task into the canonical record shape used by foreign sinks."""
    validate_mapping_fields(input_field, None)
    record: dict[str, Any] = task.model_dump(mode="json")
    record[input_field] = record.pop("input_data")
    return record


def _task_values(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key in Task.model_fields and key not in {"input_data", "metadata"}
    }


def _flat_metadata(row: Mapping[str, Any], input_field: str) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in Task.model_fields and key != input_field
    }


def _record_metadata(
    row: Mapping[str, Any],
    flat_metadata: Mapping[str, Any],
    metadata_fields: Sequence[str] | None,
) -> dict[str, Any]:
    if "metadata" in row:
        embedded_metadata = row["metadata"]
        if not isinstance(embedded_metadata, Mapping):
            raise ValueError("metadata field must be an object")
        if flat_metadata:
            raise ValueError("embedded metadata cannot be combined with flat metadata")
        return dict(embedded_metadata)
    if metadata_fields is None:
        return dict(flat_metadata)
    return {field: flat_metadata[field] for field in metadata_fields if field in flat_metadata}


def map_record(
    row: Mapping[str, Any],
    *,
    input_field: str = "input",
    metadata_fields: Sequence[str] | None = None,
) -> Task:
    """Map one foreign record into a Task with unambiguous field ownership."""
    validate_mapping_fields(input_field, metadata_fields)
    if input_field not in row:
        raise ValueError(f"missing required input field {input_field!r}")
    if "input_data" in row:
        raise ValueError(
            "input_data cannot be combined with the configured input field "
            f"{input_field!r}"
        )

    raw_input = row[input_field]
    input_data = dict(raw_input) if isinstance(raw_input, Mapping) else {"value": raw_input}
    task_values = _task_values(row)
    task_values.setdefault("name", str(raw_input)[:80] if raw_input else "unnamed")

    return Task(
        input_data=input_data,
        metadata=_record_metadata(row, _flat_metadata(row, input_field), metadata_fields),
        **task_values,
    )
