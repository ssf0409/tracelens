"""The smallest possible tracelens run — three minutes from checkout to output.

What this shows, end to end:

1. **A task** — what the agent must do, plus the expected answer.
2. **An agent** — any async callable. Wrapped via ``SimpleAdapter`` so tracelens
   can drive it.
3. **A grader** — deterministic code that scores the agent's output.
4. **A run** — three tasks, three trials each, pass/fail in milliseconds.

There is no LLM call, no network, no config file. Once you understand
this script, the other examples (``contract_eval.py``, ``http_agent_eval.py``,
``noise_aware_regression.py``) just add layers on the same skeleton.

Run it::

    python examples/hello_world.py

Then render the generated JSON report::

    tracelens report --results examples/reports/hello_world_report.json --format markdown
"""

import asyncio
import json
from pathlib import Path

from tracelens import (
    CodeGrader,
    EvalSet,
    EvaluationRunner,
    RunnerConfig,
    SimpleAdapter,
    Task,
    Transcript,
)
from tracelens.baselines.comparison import RegressionDetector, RegressionSeverity
from tracelens.baselines.manager import BaselineType, TaskBaseline
from tracelens.reporting.generator import ReportData, ReportGenerator


# ---------------------------------------------------------------------------
# 1. The agent — anything async that produces an answer.
#
# SimpleAdapter passes ``task.input_data`` straight in, so this function
# just receives a dict. A real agent would call an LLM, run tools, etc.
# The eval framework doesn't care — it just records what came back.
# ---------------------------------------------------------------------------
async def my_agent(input_data: dict) -> str:
    return str(input_data["a"] + input_data["b"])


# ---------------------------------------------------------------------------
# 2. The grader — score the answer against the expected value.
#
# CodeGrader is the deterministic flavor: same inputs → same outputs.
# (LLMGrader is the non-deterministic flavor for subjective dimensions
# like tone, helpfulness, etc.)
# ---------------------------------------------------------------------------
class ExactMatchGrader(CodeGrader):
    """Pass if final_output equals task.input_data['expected']."""

    def compute_metrics(self, transcript: Transcript, task: Task) -> dict[str, float]:
        actual = str(transcript.final_output).strip()
        expected = str(task.input_data["expected"]).strip()
        return {"exact_match": 1.0 if actual == expected else 0.0}

    def determine_pass(
        self, metrics: dict[str, float], task: Task
    ) -> tuple[bool, float]:
        score = metrics["exact_match"]
        return score == 1.0, score


# ---------------------------------------------------------------------------
# 3. Build the eval set — three tasks so you can see pass + fail in one run.
# ---------------------------------------------------------------------------
def build_eval_set() -> EvalSet:
    return EvalSet(
        name="hello-world",
        description="Trivial arithmetic eval — the smallest possible tracelens demo.",
        tasks=[
            Task(task_id="add-2-2", name="add 2+2", input_data={"a": 2, "b": 2, "expected": "4"}),
            Task(task_id="add-10-5", name="add 10+5", input_data={"a": 10, "b": 5, "expected": "15"}),
            Task(task_id="add-7-8", name="add 7+8", input_data={"a": 7, "b": 8, "expected": "15"}),
        ],
    )


def _demo_baseline(task_id: str) -> TaskBaseline:
    baseline = TaskBaseline(
        task_id=task_id,
        task_name=task_id,
        baseline_type=BaselineType.CAPABILITY,
    )
    baseline.add_metric(
        metric_name="pass_rate",
        value=1.0,
        std=0.0,
        sample_size=10,
        relative_threshold=0.1,
    )
    return baseline


def _render_sample_report(generator: ReportGenerator, report: ReportData) -> str:
    detector = RegressionDetector()
    baseline_rows: list[tuple[str, float, float, float, str, bool]] = []

    for task_summary in report.task_summaries:
        baseline = _demo_baseline(task_summary.task_id)
        baseline_metric = baseline.get_metric("pass_rate")
        if baseline_metric is None:
            continue

        regression = detector.compare(
            baseline,
            [{"pass_rate": task_summary.pass_rate, "mean_score": task_summary.mean_score}],
        )
        baseline_rows.append(
            (
                task_summary.task_id,
                baseline_metric.baseline_value,
                task_summary.pass_rate,
                task_summary.pass_rate - baseline_metric.baseline_value,
                "regression" if regression.has_regression else "no regression",
                regression.should_block_ci(RegressionSeverity.MODERATE),
            )
        )

    blocks_ci = any(row[5] for row in baseline_rows)
    markdown_report = generator.render_markdown(report)
    ci_summary = generator.render_ci_summary(report)

    lines = [
        "# TraceLens Sample Report: hello_world",
        "",
        "Generated by `python examples/hello_world.py`.",
        "",
        "## Evaluation Setup",
        "",
        f"- **Tasks**: {report.total_tasks}",
        f"- **Trials**: {report.total_trials} (3 runs per task)",
        "- **Agent**: local async arithmetic function through `SimpleAdapter`",
        "- **Baseline**: capability baseline requiring `pass_rate >= 1.0`",
        "",
        "## Graders",
        "",
        "- `hello.exact_match` (`CodeGrader`): exact string match against `task.input_data[\"expected\"]`.",
        "",
        "## CLI Markdown Output",
        "",
        markdown_report,
        "",
        "## Baseline Comparison",
        "",
        "| Task | Baseline pass_rate | Current pass_rate | Delta | Result |",
        "|------|--------------------|-------------------|-------|--------|",
    ]

    for task_id, baseline_value, current_value, delta, result, _blocks in baseline_rows:
        lines.append(
            f"| `{task_id}` | {baseline_value:.4f} | {current_value:.4f} | "
            f"{delta:+.4f} | {result} |"
        )

    lines.extend([
        "",
        "## Regression Result",
        "",
        f"- **Blocking regression**: {'yes' if blocks_ci else 'no'}",
        "- **CI threshold**: moderate",
        "- **Outcome**: CI would pass; no task regressed against the demo baseline.",
        "",
        "## CI Summary",
        "",
        "```text",
        ci_summary,
        "```",
    ])

    return "\n".join(lines)


def _relative_to_cwd(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _write_reports(report: ReportData, generator: ReportGenerator) -> tuple[Path, Path]:
    reports_dir = Path(__file__).parent / "reports"
    reports_dir.mkdir(exist_ok=True)

    json_path = reports_dir / "hello_world_report.json"
    json_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n")

    markdown_path = reports_dir / "hello_world_report.md"
    markdown_path.write_text(_render_sample_report(generator, report) + "\n")

    return json_path, markdown_path


# ---------------------------------------------------------------------------
# 4. Run it.
# ---------------------------------------------------------------------------
async def main() -> None:
    eval_set = build_eval_set()
    adapter = SimpleAdapter(my_agent)
    grader = ExactMatchGrader("hello.exact_match")

    runner = EvaluationRunner(
        adapter=adapter,
        graders=[grader],
        config=RunnerConfig(num_runs=3, max_concurrency=1),
    )
    batch = await runner.run(eval_set)
    generator = ReportGenerator(k_values=[1, 3], consistency_k_values=[2, 3])
    report = generator.build_report(batch)
    report_json, report_markdown = _write_reports(report, generator)

    print("\ntracelens hello-world")
    print("--------------------")
    print(f"trials run : {len(batch.trials)}")
    print(f"pass rate  : {batch.pass_rate:.0%}")
    print(f"report json: {_relative_to_cwd(report_json)}")
    print(f"sample md  : {_relative_to_cwd(report_markdown)}")
    print()
    for trial in batch.trials:
        output = trial.transcript.final_output if trial.transcript else None
        print(
            f"  {trial.task_id:<24s} "
            f"status={trial.status.value:<8s} "
            f"output={output!r}"
        )


if __name__ == "__main__":
    asyncio.run(main())
