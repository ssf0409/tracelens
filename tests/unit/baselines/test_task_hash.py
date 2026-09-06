"""Baselines record the content hash of the task they were stored for (issue #51)."""

import json
from pathlib import Path

from tracelens.baselines.manager import BaselineManager, TaskBaseline


def test_task_hash_is_stored_and_round_trips(tmp_path: Path) -> None:
    manager = BaselineManager(tmp_path / "b.json")
    manager.set_baseline(TaskBaseline(task_id="t", task_hash="a" * 64))
    manager.save()
    reloaded = BaselineManager(tmp_path / "b.json").get_baseline("t")
    assert reloaded is not None and reloaded.task_hash == "a" * 64


def test_legacy_baseline_files_load_with_unknown_hash(tmp_path: Path) -> None:
    path = tmp_path / "b.json"
    path.write_text(json.dumps({"t": {"task_id": "t", "metrics": {}}}))
    baseline = BaselineManager(path).get_baseline("t")
    assert baseline is not None and baseline.task_hash is None


def test_update_and_promote_record_the_hash_and_keep_it_when_omitted(tmp_path: Path) -> None:
    manager = BaselineManager(tmp_path / "b.json")
    baseline = manager.update_baseline("t", {"pass_rate": 1.0}, task_hash="a" * 64)
    assert baseline.task_hash == "a" * 64
    manager.update_baseline("t", {"pass_rate": 0.9})
    assert baseline.task_hash == "a" * 64
    baseline.promote({"pass_rate": 0.95}, task_hash="b" * 64)
    assert baseline.task_hash == "b" * 64
    baseline.promote({"pass_rate": 0.96})
    assert baseline.task_hash == "b" * 64
