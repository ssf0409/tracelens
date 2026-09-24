"""Saved `tracelens run --save-trials` artifacts for `tracelens compare` tests.

Regenerate with ``python tests/fixtures/compare/generate.py``; the derived
variants below are computed from ``baseline`` so they can never drift from it.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

FIXTURE_DIR = Path(__file__).parent
SAVED = ("baseline", "improved", "regressed", "noisy")


def load(name: str) -> dict[str, Any]:
    """The raw JSON of a saved artifact."""
    data: dict[str, Any] = json.loads((FIXTURE_DIR / f"{name}.trials.json").read_text())
    return data


def derived_identical() -> dict[str, Any]:
    """The baseline run again, as a new run: same outcomes, new run id."""
    data = copy.deepcopy(load("baseline"))
    data["batch_id"] = "identical-rerun"
    data["provenance"]["run_id"] = "identical-rerun"
    return data


def derived_edited(task_id: str = "t04") -> dict[str, Any]:
    """The baseline run of an eval set where one task's content changed."""
    data = copy.deepcopy(load("baseline"))
    measurement = data["provenance"]["measurement"]
    measurement["task_hashes"][task_id] = "e" * 64
    measurement["eval_set_hash"] = "e" * 64
    return data


def derived_legacy() -> dict[str, Any]:
    """The baseline artifact as an older TraceLens would have written it."""
    data = copy.deepcopy(load("baseline"))
    del data["provenance"]
    return data


def derived_subset(name: str, task_ids: Sequence[str]) -> dict[str, Any]:
    """A saved artifact as a run of a smaller eval set holding only ``task_ids``."""
    data = copy.deepcopy(load(name))
    keep = set(task_ids)
    data["trials"] = [trial for trial in data["trials"] if trial["task_id"] in keep]
    measurement = data["provenance"]["measurement"]
    measurement["task_hashes"] = {
        task_id: digest
        for task_id, digest in measurement["task_hashes"].items()
        if task_id in keep
    }
    measurement["eval_set_hash"] = hashlib.sha256(
        "".join(sorted(measurement["task_hashes"].values())).encode()
    ).hexdigest()
    return data
