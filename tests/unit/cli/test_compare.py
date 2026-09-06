"""`tracelens compare` over the saved fixtures (issue #28)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.fixtures.compare import (
    FIXTURE_DIR,
    derived_edited,
    derived_identical,
    derived_legacy,
    load,
)
from tracelens.cli.compare import cmd_compare
from tracelens.cli.main import build_parser
from tracelens.core.trial import TrialBatch
from tracelens.statistics.run_comparison import Verdict, compare_runs

BASELINE = FIXTURE_DIR / "baseline.trials.json"
IMPROVED = FIXTURE_DIR / "improved.trials.json"
REGRESSED = FIXTURE_DIR / "regressed.trials.json"
NOISY = FIXTURE_DIR / "noisy.trials.json"


def _run(*argv: str) -> int:
    return cmd_compare(build_parser().parse_args(["compare", *argv]))


def _batch(data: dict) -> TrialBatch:
    return TrialBatch.from_dict(data)


class TestSavedFixtures:
    """The saved real-workflow artifacts reach every verdict."""

    def test_fixtures_are_genuine_run_artifacts(self):
        for name in ("baseline", "improved", "regressed", "noisy"):
            batch = _batch(load(name))
            assert batch.provenance is not None
            assert batch.provenance.measurement.eval_set_name == "compare-fixture"
            assert len(batch.provenance.measurement.task_hashes) == 16
            assert batch.provenance.candidate.adapter.version == f"fixture-agent-{name}"

    @pytest.mark.parametrize(
        ("candidate", "verdict", "exit_code"),
        [
            ("improved", Verdict.IMPROVEMENT, 0),
            ("regressed", Verdict.REGRESSION, 1),
            ("noisy", Verdict.INCONCLUSIVE, 2),
        ],
    )
    def test_verdicts(self, candidate, verdict, exit_code):
        result = compare_runs(_batch(load("baseline")), _batch(load(candidate)))
        assert result.verdict is verdict and result.exit_code == exit_code
        assert result.alignment.compared == 16 and result.alignment.aligned_by == "content"
        assert result.compatibility.candidate_changed and result.compatibility.adapter_changed
        assert list(result.compatibility.candidate_diff) == ["prompts"]

    def test_identical_rerun_is_equivalent(self):
        result = compare_runs(_batch(load("baseline")), _batch(derived_identical()))
        assert result.verdict is Verdict.EQUIVALENT and result.exit_code == 0
        assert result.delta == 0.0 and result.p_value == 1.0
        assert "What moved: nothing" in "\n".join(result.summary_lines())

    def test_latency_is_lower_is_better(self):
        result = compare_runs(
            _batch(load("baseline")), _batch(load("improved")),
            metric="fixture.latency_ms", direction="lower", threshold=50.0,
        )
        assert result.verdict is Verdict.IMPROVEMENT
        assert result.raw_delta == pytest.approx(-200.0) and result.delta == pytest.approx(200.0)
        slower = compare_runs(
            _batch(load("baseline")), _batch(load("regressed")),
            metric="fixture.latency_ms", direction="lower", threshold=50.0,
        )
        assert slower.verdict is Verdict.REGRESSION

    def test_grader_filter_matches_the_single_grader_run(self):
        both = compare_runs(_batch(load("baseline")), _batch(load("improved")))
        only = compare_runs(_batch(load("baseline")), _batch(load("improved")), grader="fixture")
        assert only.delta == both.delta and only.grader == "fixture"


class TestCommand:
    def test_exit_codes_and_summary(self, capsys: pytest.CaptureFixture[str]):
        assert _run(str(BASELINE), str(IMPROVED)) == 0
        out = capsys.readouterr().out
        assert out.startswith("Compared improved.trials.json vs baseline.trials.json on pass_rate")
        assert "Verdict: IMPROVEMENT (exit 0)" in out
        assert "What changed: adapter, DecisionSpec prompts" in out
        assert _run(str(BASELINE), str(REGRESSED)) == 1
        assert "Verdict: REGRESSION (exit 1)" in capsys.readouterr().out
        assert _run(str(BASELINE), str(NOISY)) == 2
        assert "inconclusive" in capsys.readouterr().out

    def test_observe_mode_exits_zero(self, capsys: pytest.CaptureFixture[str]):
        assert _run(str(BASELINE), str(NOISY), "--observe") == 0
        assert "inconclusive" in capsys.readouterr().out

    def test_output_json_shares_the_summary_facts(self, tmp_path: Path, capsys):
        target = tmp_path / "out" / "compare.json"
        assert _run(str(BASELINE), str(IMPROVED), "--output", str(target), "--seed", "3") == 0
        captured = capsys.readouterr()
        data = json.loads(target.read_text())
        assert data["method"] == "paired task bootstrap" and data["unit"] == "task"
        assert data["verdict"] == "improvement" and data["exit_code"] == 0
        assert data["seed"] == 3 and data["n_bootstrap"] == 10000 and data["confidence"] == 0.95
        assert f"delta = {data['delta']:+.4f}" in captured.out
        assert f"[{data['ci_lower']:+.4f}, {data['ci_upper']:+.4f}]" in captured.out
        assert data["per_task"][0]["task_id"] in captured.out
        assert data["compatibility"]["status"] == "compatible"
        assert f"[tracelens] wrote comparison: {target}" in captured.err

    def test_edited_task_is_refused_then_excluded(self, tmp_path: Path, capsys):
        edited = tmp_path / "edited.trials.json"
        edited.write_text(json.dumps(derived_edited()))
        assert _run(str(BASELINE), str(edited)) == 2
        assert "task sets differ: 1 changed content (t04)" in capsys.readouterr().err
        assert _run(str(BASELINE), str(edited), "--unmatched-tasks", "exclude") == 0
        out = capsys.readouterr().out
        assert "15 task(s) compared, aligned by content; 1 excluded (changed content: t04)" in out
        assert "equivalent within the practical threshold" in out

    def test_legacy_artifact_is_labelled_or_refused(self, tmp_path: Path, capsys):
        legacy = tmp_path / "legacy.trials.json"
        legacy.write_text(json.dumps(derived_legacy()))
        assert _run(str(legacy), str(IMPROVED)) == 0
        out = capsys.readouterr().out
        assert "aligned by id" in out and "What changed: unknown" in out
        assert _run(str(legacy), str(IMPROVED), "--require-provenance") == 2
        assert "drop --require-provenance" in capsys.readouterr().err

    @pytest.mark.parametrize(
        ("argv", "fragment"),
        [
            (["missing.json", str(IMPROVED)], "trials file not found"),
            ([str(BASELINE), str(IMPROVED), "--metric", "speed"], "unknown metric 'speed'"),
            ([str(BASELINE), str(IMPROVED), "--metric", "pass_rate", "--direction", "lower"],
             "always higher-is-better"),
            ([str(BASELINE), str(IMPROVED), "--threshold", "-1"], "threshold cannot be negative"),
        ],
    )
    def test_input_errors_exit_2_without_output(self, argv, fragment, capsys):
        assert _run(*argv) == 2
        captured = capsys.readouterr()
        assert fragment in captured.err and captured.out == ""

    def test_results_file_and_invalid_json_are_rejected(self, tmp_path: Path, capsys):
        results = tmp_path / "results.json"
        results.write_text(json.dumps({"total_trials": 1, "total_tasks": 1, "task_summaries": []}))
        assert _run(str(results), str(IMPROVED)) == 2
        err = capsys.readouterr().err
        assert "is a results file" in err and "--save-trials" in err
        broken = tmp_path / "broken.json"
        broken.write_text("{not json")
        assert _run(str(BASELINE), str(broken)) == 2
        assert "invalid JSON" in capsys.readouterr().err
        not_trials = tmp_path / "list.json"
        not_trials.write_text(json.dumps({"trials": "nope"}))
        assert _run(str(BASELINE), str(not_trials)) == 2
        assert "not a valid trials file" in capsys.readouterr().err

    def test_real_process_stdout_is_the_summary_only(self, tmp_path: Path):
        target = tmp_path / "compare.json"
        result = subprocess.run(
            [sys.executable, "-m", "tracelens.cli.main", "compare", str(BASELINE), str(REGRESSED),
             "--output", str(target), "--top", "2"],
            cwd=Path(__file__).resolve().parents[3], capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 1, result.stdout + result.stderr
        assert "Traceback" not in result.stderr
        assert result.stdout.startswith("Compared regressed.trials.json vs baseline.trials.json")
        assert "Verdict: REGRESSION (exit 1)" in result.stdout
        assert result.stderr.strip() == f"[tracelens] wrote comparison: {target}"
        assert json.loads(target.read_text())["verdict"] == "regression"
