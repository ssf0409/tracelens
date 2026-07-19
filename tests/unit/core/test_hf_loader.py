"""Contract tests for the optional Hugging Face Task loader."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest

import tracelens
import tracelens.loaders as loaders
from tracelens.core.task import Task
from tracelens.loaders import HFDatasetLoader


class _FakeDatasetDict(dict[str, list[dict[str, Any]]]):
    """Minimal DatasetDict stand-in for split-selection tests."""


def _fake_datasets_module(
    *,
    hub_rows: list[Any] | None = None,
    local_dataset: Any = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        Dataset=SimpleNamespace(from_list=Mock()),
        DatasetDict=_FakeDatasetDict,
        load_dataset=Mock(return_value=[] if hub_rows is None else hub_rows),
        load_from_disk=Mock(return_value=local_dataset),
    )


def test_hf_loader_is_subpackage_only_public_api() -> None:
    assert "HFDatasetLoader" in loaders.__all__
    assert not hasattr(tracelens, "HFDatasetLoader")


def test_missing_datasets_extra_has_actionable_error() -> None:
    missing = ModuleNotFoundError("No module named 'datasets'", name="datasets")

    with (
        patch("tracelens.loaders.hf.import_module", side_effect=missing),
        pytest.raises(ImportError, match=r"tracelens\[datasets\]"),
    ):
        HFDatasetLoader(split="test").load("owner/dataset")


def test_missing_transitive_dependency_is_not_misreported() -> None:
    missing = ModuleNotFoundError("No module named 'pyarrow'", name="pyarrow")

    with (
        patch("tracelens.loaders.hf.import_module", side_effect=missing),
        pytest.raises(ModuleNotFoundError, match="pyarrow"),
    ):
        HFDatasetLoader(split="test").load("owner/dataset")


def test_hub_source_requires_an_explicit_split() -> None:
    with pytest.raises(ValueError, match="split is required"):
        HFDatasetLoader().load("owner/dataset")


def test_hub_load_forwards_reproducibility_options_and_maps_rows() -> None:
    module = _fake_datasets_module(
        hub_rows=[
            {
                "question": "What is 2 + 2?",
                "name": "Arithmetic",
                "subject": "math",
                "private": "ignored",
            }
        ]
    )
    loader = HFDatasetLoader(
        input_field="question",
        metadata_fields=["subject"],
        config_name="all",
        split="test",
        revision="abc123",
    )

    with patch("tracelens.loaders.hf._require_datasets", return_value=module):
        tasks = loader.load("owner/dataset")

    module.load_dataset.assert_called_once_with(
        "owner/dataset",
        name="all",
        split="test",
        revision="abc123",
    )
    assert len(tasks) == 1
    assert tasks[0].name == "Arithmetic"
    assert tasks[0].input_data == {"value": "What is 2 + 2?"}
    assert tasks[0].metadata == {"subject": "math"}


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ({"name": "missing input"}, "missing required input field"),
        (["not", "a", "mapping"], "row must be a mapping"),
    ],
)
def test_hub_row_errors_include_dataset_split_and_row_number(
    row: Any,
    message: str,
) -> None:
    module = _fake_datasets_module(hub_rows=[{"input": "valid"}, row])

    with (
        patch("tracelens.loaders.hf._require_datasets", return_value=module),
        pytest.raises(ValueError, match=rf"owner/dataset:test row 2: {message}"),
    ):
        HFDatasetLoader(split="test").load("owner/dataset")


def test_local_dataset_path_loads_without_hub_dispatch(tmp_path: Path) -> None:
    source = tmp_path / "saved-dataset"
    module = _fake_datasets_module(local_dataset=[{"prompt": {"goal": "local"}, "name": "Local"}])

    with patch("tracelens.loaders.hf._require_datasets", return_value=module):
        tasks = HFDatasetLoader(input_field="prompt").load(source)

    module.load_from_disk.assert_called_once_with(str(source))
    module.load_dataset.assert_not_called()
    assert tasks[0].input_data == {"goal": "local"}


def test_local_dataset_dict_requires_a_split(tmp_path: Path) -> None:
    module = _fake_datasets_module(local_dataset=_FakeDatasetDict(train=[{"input": "train row"}]))

    with (
        patch("tracelens.loaders.hf._require_datasets", return_value=module),
        pytest.raises(ValueError, match="split is required.*train"),
    ):
        HFDatasetLoader().load(tmp_path / "saved-dataset-dict")


def test_local_dataset_dict_rejects_an_unknown_split(tmp_path: Path) -> None:
    module = _fake_datasets_module(local_dataset=_FakeDatasetDict(train=[{"input": "train row"}]))

    with (
        patch("tracelens.loaders.hf._require_datasets", return_value=module),
        pytest.raises(ValueError, match="split 'test' not found.*train"),
    ):
        HFDatasetLoader(split="test").load(tmp_path / "saved-dataset-dict")


def test_local_dataset_dict_selects_the_configured_split(tmp_path: Path) -> None:
    module = _fake_datasets_module(
        local_dataset=_FakeDatasetDict(
            train=[{"input": "train row"}],
            test=[{"input": "test row", "name": "Test"}],
        )
    )

    with patch("tracelens.loaders.hf._require_datasets", return_value=module):
        tasks = HFDatasetLoader(split="test").load(tmp_path / "saved-dataset-dict")

    assert [task.name for task in tasks] == ["Test"]


def test_save_uses_canonical_records_and_creates_the_destination_parent(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "nested" / "saved-dataset"
    saved_dataset = Mock()
    module = _fake_datasets_module()
    module.Dataset.from_list.return_value = saved_dataset
    task = Task(
        task_id="hf-save",
        name="Saved",
        input_data={"goal": "persist"},
        metadata={"source": "unit"},
    )

    with patch("tracelens.loaders.hf._require_datasets", return_value=module):
        HFDatasetLoader(input_field="prompt").save([task], destination)

    records = module.Dataset.from_list.call_args.args[0]
    assert records[0]["prompt"] == {"goal": "persist"}
    assert "input_data" not in records[0]
    assert records[0]["metadata"] == {"source": "unit"}
    assert destination.parent.is_dir()
    saved_dataset.save_to_disk.assert_called_once_with(str(destination))


def test_save_rejects_an_empty_task_list(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot save an empty Task list"):
        HFDatasetLoader().save([], tmp_path / "empty-dataset")


def test_hf_loader_reuses_shared_field_ownership_validation() -> None:
    with pytest.raises(ValueError, match="conflicts with a Task field"):
        HFDatasetLoader(input_field="metadata")
