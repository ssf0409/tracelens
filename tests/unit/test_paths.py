"""Tests for shared filesystem path policy."""

from pathlib import Path

from tracelens._paths import prepare_destination_path


def test_prepare_destination_path_creates_only_the_parent(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "output.jsonl"

    prepared = prepare_destination_path(destination)

    assert prepared == destination
    assert destination.parent.is_dir()
    assert not destination.exists()

    assert prepare_destination_path(destination) == destination
