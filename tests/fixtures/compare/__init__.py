"""Saved `tracelens run --save-trials` artifacts for `tracelens compare` tests.

Regenerate with ``python tests/fixtures/compare/generate.py``; the derived
variants below are computed from ``baseline`` so they can never drift from it.
"""

from __future__ import annotations

import copy
import json
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
