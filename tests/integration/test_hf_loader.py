"""Integration coverage for the optional Hugging Face datasets dependency."""

from pathlib import Path

import pytest

from tracelens.core.task import Task, TaskExpectation
from tracelens.loaders import HFDatasetLoader

datasets = pytest.importorskip("datasets", reason="requires the datasets extra")


def test_hf_local_dataset_round_trip(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "hf-dataset"
    original = Task(
        task_id="hf-round-trip",
        name="HF round trip",
        description="Real datasets.Dataset serialization",
        input_data={"question": "What is 2 + 2?"},
        expectation=TaskExpectation(expected_output="4"),
        metadata={"source": "integration"},
        tags=["math"],
        difficulty="easy",
        category="reasoning",
        timeout_seconds=10.0,
    )
    loader = HFDatasetLoader(input_field="prompt")

    loader.save([original], destination)
    loaded = loader.load(destination)

    assert loaded == [original]
    assert datasets.load_from_disk(str(destination)).column_names == [
        "task_id",
        "name",
        "description",
        "expectation",
        "metadata",
        "tags",
        "difficulty",
        "category",
        "timeout_seconds",
        "prompt",
    ]
