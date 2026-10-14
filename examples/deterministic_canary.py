"""Frozen replay -> transcript -> metrics -> persisted baseline -> strict gate.

Record once, then check candidates against that reference. Every check writes
trials.json and gate.json and leaves the baseline unchanged. The changes are
deliberate fault injections into a local workflow, with no network or LLM.
This is a downstream Python policy, separate from the statistical CLI gate.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from tracelens import (
    CodeGrader,
    EvalSet,
    EvaluationRunner,
    RunnerConfig,
    SimpleAdapter,
    Task,
    Transcript,
)
from tracelens.baselines import BaselineManager, TaskBaseline
from tracelens.core.decision_spec import DecisionSpec
from tracelens.core.provenance import task_content_hash

REPLAY = Path(__file__).with_name("canary_replay.json")
SCENARIOS = ("unchanged", "increase", "decrease", "missing", "new", "empty", "zero-channel")


async def replay(input_data: dict[str, Any], scenario: str) -> dict[str, Any]:
    decisions = [dict(request) for request in input_data["requests"]]
    if scenario == "increase":
        decisions.append(dict(decisions[0]))  # Duplicate-processing fault.
    elif scenario == "decrease":
        decisions.pop(0)  # Dropped-request fault.
    if scenario == "new":
        # Unknown-reason fault: an added rejection, so no known count moves.
        decisions.append({"id": "d", "approved": False, "reason": "permission"})
    elif scenario == "zero-channel":
        # A known counter recorded at zero activates; no other count moves.
        decisions.append({"id": "d", "approved": False, "reason": "quota"})
    return {
        "decisions": decisions,
        "emit_accepted": scenario != "missing",
        "emit_metrics": scenario != "empty",
    }


class WorkflowMetrics(CodeGrader):
    """Extract observed counts; the downstream canary decides whether they moved."""

    def compute_metrics(self, transcript: Transcript, task: Task) -> dict[str, float]:
        output = transcript.final_output
        if not isinstance(output, dict):
            raise ValueError("Workflow output must be a dictionary")
        if not output["emit_metrics"]:
            return {}
        metrics = {f"rejected.{reason}": 0.0 for reason in task.input_data["known_reasons"]}
        accepted = 0.0
        for decision in output["decisions"]:
            if decision["approved"]:
                accepted += 1
            else:
                name = f"rejected.{decision['reason']}"
                metrics[name] = metrics.get(name, 0.0) + 1
        if output["emit_accepted"]:
            metrics["accepted"] = accepted
        return metrics

    def determine_pass(self, metrics: dict[str, float], task: Task) -> tuple[bool, float]:
        # Producing well-formed metrics is distinct from matching the frozen reference.
        return True, 1.0


async def run(args: argparse.Namespace) -> int:
    baseline_path: Path = args.baseline
    output_dir: Path = args.output_dir
    artifact_paths = [output_dir / "trials.json", output_dir / "gate.json"]
    if baseline_path.resolve() in {path.resolve() for path in artifact_paths}:
        raise ValueError("Baseline must be separate from output artifacts")
    if any(path.exists() for path in artifact_paths):
        raise ValueError("Use a fresh output directory to preserve prior evidence")
    if args.mode == "record" and baseline_path.exists():
        raise ValueError("Baseline already exists; recording cannot replace it")
    if args.mode == "check" and not baseline_path.is_file():
        raise ValueError("Baseline file does not exist; record the reference first")
    if args.mode == "check":
        stored = json.loads(baseline_path.read_text())
        if not isinstance(stored, dict):
            raise ValueError("Baseline file must contain a task mapping")
        for entry in stored.values():
            TaskBaseline.model_validate(entry)

    eval_set = EvalSet.model_validate(json.loads(REPLAY.read_text()))
    task = eval_set.tasks[0]
    task_hash = task_content_hash(task)
    manager = BaselineManager(baseline_path)
    baseline = manager.get_baseline(task.task_id)
    if args.mode == "check":
        if baseline is None or not baseline.metrics:
            raise ValueError("No populated baseline for this task")
        if baseline.task_hash != task_hash:
            raise ValueError("Frozen task changed; compare only the recorded task")
        # A metric with no tolerance has no `changed` verdict, so it could move
        # freely: recorded before tolerances existed, or added by promotion.
        unprotected = sorted(
            name for name, metric in baseline.metrics.items()
            if metric.change_tolerance_absolute is None
            and metric.change_tolerance_relative is None
        )
        if unprotected:
            raise ValueError(
                f"Metrics without a change tolerance: {', '.join(unprotected)}; "
                "review them and set one before checking"
            )

    async def candidate(input_data: dict[str, Any]) -> dict[str, Any]:
        return await replay(input_data, args.scenario)

    spec = DecisionSpec(extra={"scenario": args.scenario})
    runner = EvaluationRunner(
        SimpleAdapter(candidate), [WorkflowMetrics("workflow-counts")],
        RunnerConfig(num_runs=1), decision_spec=spec,
    )
    batch = await runner.run(eval_set)
    if batch.gradable_count != 1 or len(batch.trials[0].outcomes) != 1:
        raise ValueError("Replay did not produce one gradable measurement")
    metrics = batch.trials[0].outcomes[0].metrics
    if args.mode == "record":
        if not metrics:
            raise ValueError("Cannot record an empty measurement set")
        baseline = manager.create_canary_baseline(task.task_id, metrics, decision_spec=spec)
        for metric in baseline.metrics.values():
            # Assign, not add_metric(): that would replace the whole metric.
            metric.change_tolerance_absolute = 0.0
        baseline.task_hash = task_hash
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        manager.save()
        # Check the serialized reference, not only the in-memory object.
        manager = BaselineManager(baseline_path)

    comparisons = manager.compare_to_baseline(task.task_id, metrics, coverage="strict")
    blocked = not comparisons or any(
        row["coverage"] != "matched" or row["regression"] or row.get("changed", True)
        for row in comparisons.values()
    )
    exit_code = 1 if blocked else 0
    gate = {
        "status": "blocked" if blocked else "passed", "exit_code": exit_code,
        "coverage": "strict", "scenario": args.scenario, "task_hash": task_hash,
        "metrics": metrics, "comparisons": comparisons,
        "run_id": batch.provenance.run_id if batch.provenance else None,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths[0].write_text(json.dumps(batch.to_dict(), indent=2, allow_nan=False) + "\n")
    artifact_paths[1].write_text(json.dumps(gate, indent=2, allow_nan=False) + "\n")
    print(json.dumps(gate, allow_nan=False))
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("record", "check"))
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, default="unchanged")
    args = parser.parse_args()
    if args.mode == "record" and args.scenario != "unchanged":
        parser.error("Record only the unchanged reference")
    try:
        return asyncio.run(run(args))
    except (ValueError, OSError) as exc:
        print(f"canary: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
