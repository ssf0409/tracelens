"""Infrastructure-noise-aware regression detection in 80 lines.

This is the differentiator. Anthropic's "Quantifying infrastructure
noise in agentic coding evals" (Feb 2026) showed that resource-config
changes alone can swing Terminal-Bench scores by up to 6pp — often
more than the leaderboard gap between frontier models. tracelens makes
that effect first-class: runs are fingerprinted including their
`InfraConfig`, and the regression detector flags sub-3pp deltas as
"within-noise-band" when the two runs' configs don't match.

Three moving pieces:
- `InfraError` raised by your adapter when a failure is known to be
  infrastructure (OOM, sandbox killed). The runner classifies that
  separately from task failure.
- `DecisionSpec.infra` = `InfraConfig(...)` makes the resource config
  part of the fingerprint. Two runs with different configs collide to
  DIFFERENT fingerprints; two runs with identical configs on different
  hosts collide to the SAME fingerprint (observational fields like
  hostname are excluded from the hash — this is deliberate).
- `RegressionDetector.compare_with_specs()` takes both specs, flags
  infra mismatch, and marks sub-noise-band regressions as
  `within_noise_band=True` so default CI gates don't block on them.

Run: `python examples/noise_aware_regression.py`
"""

import asyncio

from tracelens import (
    DecisionSpec,
    EvalSet,
    EvaluationRunner,
    InfraConfig,
    InfraError,
    RegressionDetector,
    RunnerConfig,
    SimpleAdapter,
    Task,
    TaskBaseline,
)


async def flaky_agent(input_data: dict) -> dict:
    """Fails when memory budget < declared need — simulating OOM kill."""
    budget = input_data.get("_budget_mb", 2048)
    needed = input_data.get("memory_mb_needed", 0)
    if needed > budget:
        raise InfraError(f"OOM: needed {needed}MB, budget {budget}MB")
    return {"ok": True}


async def run_once(budget_mb: int) -> tuple[float, float, DecisionSpec]:
    spec = DecisionSpec(infra=InfraConfig(
        memory_hard_limit_mb=budget_mb,
        runtime_platform="local-demo",
    ))
    tasks = [
        Task(name="lean", input_data={"_budget_mb": budget_mb, "memory_mb_needed": 128}),
        Task(name="heavy", input_data={"_budget_mb": budget_mb, "memory_mb_needed": 1024}),
    ]
    batch = await EvaluationRunner(
        SimpleAdapter(flaky_agent), [], RunnerConfig(num_runs=3),
    ).run(EvalSet(name="demo", tasks=tasks))
    # Pass rate from the runner: INFRA_ERROR trials don't count as passes.
    # NOTE: the CLI baseline gate EXCLUDES infra-error trials from its
    # comparison samples (they surface via infra_error_rate instead); this
    # example folds them into the rate deliberately to show the raw score
    # swing an infra change can cause.
    completed_rate = sum(1 for t in batch.trials if t.is_successful) / len(batch.trials)
    return completed_rate, batch.infra_error_rate, spec


async def main() -> None:
    # --- 1. Record a baseline under generous memory. ---
    baseline_rate, baseline_infra_err, baseline_spec = await run_once(budget_mb=2048)
    print(f"Baseline (2048MB): pass={baseline_rate:.0%}  infra_errors={baseline_infra_err:.0%}")

    # --- 2. Re-run under tight memory. ---
    current_rate, current_infra_err, current_spec = await run_once(budget_mb=512)
    print(f"Current  ( 512MB): pass={current_rate:.0%}  infra_errors={current_infra_err:.0%}")

    # --- 3. Compare with spec awareness. ---
    baseline = TaskBaseline(task_id="demo")
    baseline.add_metric("pass_rate", value=baseline_rate, std=0.02, sample_size=6)
    report = RegressionDetector(min_delta_percent=1.0).compare_with_specs(
        baseline,
        current_results=[{"pass_rate": current_rate}] * 6,
        baseline_spec=baseline_spec,
        current_spec=current_spec,
    )

    print(f"\nInfra config mismatch: {report.infra_config_mismatch}")
    if report.infra_config_mismatch:
        for key, (was, now) in sorted(report.infra_config_diff.items()):
            print(f"  {key}: {was!r} -> {now!r}")
    print(f"Regressions found:      {len(report.regressions)}")
    print(f"Blocking (post-filter): {len(report.blocking_regressions)}")
    print(f"should_block_ci():      {report.should_block_ci()}")


if __name__ == "__main__":
    asyncio.run(main())
