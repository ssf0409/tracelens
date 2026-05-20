"""Runner for the high-stakes-autonomous benchmark pack.

End-to-end demonstration of tracelens's infra-noise awareness.

Typical session (see README.md for the full narrative):

    # 1. Record a baseline under a generous 2GB memory cap.
    python -m benchmarks.high_stakes_autonomous.run_benchmark \
        --memory-hard-limit-mb 2048 \
        --save-baseline benchmarks/high-stakes-autonomous/baseline.json

    # 2. Re-run under a tight 512MB cap. The pass rate drops, but the
    #    regression report flags the sub-3pp deltas as within the infra-
    #    noise band because the DecisionSpec.infra section changed.
    python -m benchmarks.high_stakes_autonomous.run_benchmark \
        --memory-hard-limit-mb 512 \
        --compare-baseline benchmarks/high-stakes-autonomous/baseline.json

Argument reference:
    --memory-hard-limit-mb  Simulated container memory ceiling. Tasks
                            that declare a higher requirement will OOM
                            and produce INFRA_ERROR trials.
    --num-runs              How many trials per task (pass@k shape).
    --save-baseline PATH    Record current run's per-task pass_rate as
                            a TaskBaseline JSON blob.
    --compare-baseline PATH Load a baseline JSON and call
                            RegressionDetector.compare_with_specs so
                            the infra-config mismatch warning fires.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import socket
import sys
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASKS_PATH = HERE / "tasks.json"


# The directory uses a hyphen ("high-stakes-autonomous") which is a valid
# directory name but not a valid Python module name. Load the sibling
# modules directly by file path so `python run_benchmark.py` works without
# requiring the user to set up any package aliases or PYTHONPATH hacks.
def _load_sibling(module_name: str):
    full = f"_hsab_{module_name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, HERE / f"{module_name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod


_agent_mod = _load_sibling("agent")
_graders_mod = _load_sibling("graders")
BudgetedAgent = _agent_mod.BudgetedAgent
SafetyGateGrader = _graders_mod.SafetyGateGrader
CapabilityGrader = _graders_mod.CapabilityGrader

from tracelens.baselines.comparison import RegressionDetector  # noqa: E402
from tracelens.baselines.manager import TaskBaseline  # noqa: E402
from tracelens.core.decision_spec import DecisionSpec, InfraConfig  # noqa: E402
from tracelens.core.task import EvalSet, JSONTaskLoader, Task  # noqa: E402
from tracelens.core.trial import TrialBatch, TrialStatus  # noqa: E402
from tracelens.execution.runner import EvaluationRunner, RunnerConfig  # noqa: E402
from tracelens.reporting.generator import ReportGenerator  # noqa: E402


def _build_decision_spec(memory_hard_limit_mb: int, concurrency: int) -> DecisionSpec:
    return DecisionSpec(
        infra=InfraConfig(
            memory_guaranteed_mb=min(memory_hard_limit_mb // 2, 512),
            memory_hard_limit_mb=memory_hard_limit_mb,
            concurrency_level=concurrency,
            runtime_platform="local-mock",
            sandbox_provider="in-process",
            harness_version="tracelens-0.1.0",
            hostname=socket.gethostname(),
            wall_clock_start_utc=datetime.now(UTC),
        ),
    )


def _build_task_baseline(task_id: str, pass_rate: float) -> TaskBaseline:
    baseline = TaskBaseline(task_id=task_id)
    baseline.add_metric(
        metric_name="pass_rate",
        value=pass_rate,
        std=0.02,
        sample_size=5,
        higher_is_better=True,
    )
    return baseline


def _save_baseline(path: Path, batch: TrialBatch, spec: DecisionSpec) -> None:
    pass_rates = {
        task_id: sum(passes) / len(passes) if passes else 0.0
        for task_id, passes in batch.get_pass_results_by_task().items()
    }
    payload = {
        "decision_spec": spec.model_dump(mode="json"),
        "task_pass_rates": pass_rates,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def _load_baseline(path: Path) -> tuple[DecisionSpec, dict[str, float]]:
    data = json.loads(path.read_text())
    spec = DecisionSpec.model_validate(data["decision_spec"])
    return spec, data["task_pass_rates"]


def _print_header(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


async def _run(eval_set: EvalSet, spec: DecisionSpec, num_runs: int) -> TrialBatch:
    assert spec.infra is not None
    agent = BudgetedAgent(memory_hard_limit_mb=spec.infra.memory_hard_limit_mb or 2048)
    runner = EvaluationRunner(
        agent,
        [SafetyGateGrader(), CapabilityGrader()],
        RunnerConfig(num_runs=num_runs, max_concurrency=4),
    )
    batch = await runner.run(eval_set)
    # Attach the decision spec to every transcript so downstream tooling
    # can see which config produced which result.
    for trial in batch.trials:
        if trial.transcript is not None and trial.transcript.decision_spec is None:
            trial.transcript.decision_spec = spec
    return batch


def _compare_with_baseline(
    baseline_spec: DecisionSpec,
    baseline_pass_rates: dict[str, float],
    current_batch: TrialBatch,
    current_spec: DecisionSpec,
) -> None:
    detector = RegressionDetector(min_delta_percent=1.0)
    current_pass_results = current_batch.get_pass_results_by_task()

    print("\nRegression comparison (per-task):")
    any_infra_mismatch = False
    any_within_band = False
    for task_id, passes in sorted(current_pass_results.items()):
        if task_id not in baseline_pass_rates:
            continue
        baseline = _build_task_baseline(task_id, baseline_pass_rates[task_id])
        current_rate = sum(passes) / len(passes) if passes else 0.0
        report = detector.compare_with_specs(
            baseline,
            current_results=[{"pass_rate": current_rate}] * len(passes),
            baseline_spec=baseline_spec,
            current_spec=current_spec,
        )
        baseline_rate = baseline_pass_rates[task_id]
        delta = current_rate - baseline_rate
        flag = ""
        if report.has_regression:
            reg = report.regressions[0]
            if reg.within_noise_band:
                flag = "  [within-noise-band ⚠]"
                any_within_band = True
            else:
                flag = f"  [REGRESSION {reg.severity.value.upper()}]"
        if report.infra_config_mismatch:
            any_infra_mismatch = True
        print(
            f"  {task_id:<40s} "
            f"baseline={baseline_rate:.1%}  current={current_rate:.1%}  "
            f"Δ={delta:+.1%}{flag}"
        )
    if any_infra_mismatch:
        print(
            "\n⚠ Infra config mismatch detected between baseline and current run.\n"
            "  Regressions within the 3pp noise band have been flagged but NOT\n"
            "  counted as blocking — rerun under the baseline's resource config\n"
            "  to confirm whether the capability really dropped.\n"
            "  (Anthropic's 'Quantifying infrastructure noise in agentic coding\n"
            "  evals' recommends this; <3pp deltas under mismatched configs are\n"
            "  within the noise band.)"
        )
    if any_within_band:
        print(
            "  At least one regression was downgraded by the noise band. CI would"
            " not block by default."
        )


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-hard-limit-mb", type=int, default=2048)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--save-baseline", type=Path, default=None)
    parser.add_argument("--compare-baseline", type=Path, default=None)
    args = parser.parse_args()

    tasks: list[Task] = JSONTaskLoader().load(TASKS_PATH)
    eval_set = EvalSet(name="high-stakes-autonomous", tasks=tasks)

    current_spec = _build_decision_spec(args.memory_hard_limit_mb, args.concurrency)
    assert current_spec.infra is not None

    _print_header(
        f"Running high-stakes-autonomous benchmark  "
        f"(memory_hard_limit_mb={current_spec.infra.memory_hard_limit_mb}, "
        f"fingerprint={current_spec.fingerprint_short})"
    )

    batch = await _run(eval_set, current_spec, args.num_runs)

    gen = ReportGenerator()
    report = gen.build_report(batch)

    print(gen.render_markdown(report))
    print("--- CI Summary ---")
    print(gen.render_ci_summary(report))

    # Break down INFRA_ERROR vs FAILED for each task.
    print("\nStatus breakdown (per trial):")
    for trial in batch.trials:
        status_tag = {
            TrialStatus.COMPLETED: "pass" if trial.passed else "fail",
            TrialStatus.FAILED: "FAILED",
            TrialStatus.INFRA_ERROR: "INFRA_ERROR",
            TrialStatus.TIMEOUT: "timeout",
        }.get(trial.status, trial.status.value)
        suffix = f" — {trial.error_message}" if trial.error_message else ""
        print(f"  [{status_tag:<12s}] {trial.task_id}  run={trial.run_index}{suffix}")

    if args.save_baseline:
        _save_baseline(args.save_baseline, batch, current_spec)
        print(f"\nBaseline saved to {args.save_baseline}")

    if args.compare_baseline and args.compare_baseline.exists():
        baseline_spec, baseline_pass_rates = _load_baseline(args.compare_baseline)
        _compare_with_baseline(baseline_spec, baseline_pass_rates, batch, current_spec)

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
