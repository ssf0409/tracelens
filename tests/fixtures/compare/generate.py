"""Regenerate the saved run artifacts used by the `tracelens compare` tests.

    uv run --frozen python tests/fixtures/compare/generate.py

Every scenario runs the same twelve-task eval set through the real
``EvaluationRunner`` with a deterministic simulated agent, so the files are
genuine ``tracelens run --save-trials`` artifacts (provenance included):

- ``baseline.trials.json``   per-task pass probability spread from 0.2 to 0.8,
                             six runs per task
- ``improved.trials.json``   every task +0.3 (capped) and 200 ms faster
- ``regressed.trials.json``  every task -0.3 (floored) and 200 ms slower
- ``noisy.trials.json``      the baseline's probabilities, fresh draws, only two
                             runs per task: too little evidence for a verdict

The tests derive the remaining cases from ``baseline`` (an identical rerun,
an edited task, an artifact without provenance) in ``tests/fixtures/compare``.

Each scenario declares its own prompt version in a ``DecisionSpec`` and its
own adapter ``provenance_version``, so a comparison can say what changed.
Outcomes are decided by hashing (scenario, task, run), so regenerating
changes only run ids and timestamps.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

from tracelens.core.decision_spec import DecisionSpec, PromptSpec
from tracelens.core.grader import CodeGrader
from tracelens.core.task import EvalSet, Task
from tracelens.core.transcript import Transcript
from tracelens.execution.agent_adapter import AgentAdapter
from tracelens.execution.runner import EvaluationRunner, RunnerConfig

HERE = Path(__file__).parent
TASK_COUNT = 16
RUNS = 6


def tasks() -> EvalSet:
    return EvalSet(name="compare-fixture", tasks=[
        Task(
            task_id=f"t{i:02d}", name=f"task {i}",
            input_data={"prompt": f"question {i}", "index": i},
            difficulty=("easy", "medium", "hard")[i % 3],
        )
        for i in range(TASK_COUNT)
    ])


def base_probability(index: int) -> float:
    return 0.2 + 0.6 * index / (TASK_COUNT - 1)


class SimulatedAgent(AgentAdapter):
    """Passes each (task, run) with a per-task probability, decided by a hash."""

    def __init__(self, scenario: str, shift: float, latency_shift_ms: float) -> None:
        self.provenance_version = f"fixture-agent-{scenario}"
        self.scenario = scenario
        self.shift = shift
        self.latency_shift_ms = latency_shift_ms
        self.calls: dict[str, int] = {}

    async def run(self, task: Task) -> Transcript:
        run_index = self.calls.get(task.task_id, 0)
        self.calls[task.task_id] = run_index + 1
        index = int(task.input_data["index"])
        draw = int(hashlib.sha256(
            f"{self.scenario}-{task.task_id}-{run_index}".encode()
        ).hexdigest(), 16) / 16**64
        probability = min(1.0, max(0.0, base_probability(index) + self.shift))
        passed = draw < probability
        latency = 1000.0 + 50.0 * index + self.latency_shift_ms + (run_index % 3) * 10.0
        return Transcript(
            task_id=task.task_id,
            final_output={"answer": "right" if passed else "wrong", "latency_ms": latency},
        )


class FixtureGrader(CodeGrader):
    provenance_version = "fixture-grader-1"

    def __init__(self) -> None:
        super().__init__("fixture")

    def compute_metrics(self, transcript: Transcript, task: Task) -> dict[str, float]:
        answer = transcript.final_output["answer"]
        return {
            "correct": 1.0 if answer == "right" else 0.0,
            "latency_ms": float(transcript.final_output["latency_ms"]),
        }

    def determine_pass(self, metrics: dict[str, float], task: Task) -> tuple[bool, float]:
        return metrics["correct"] == 1.0, metrics["correct"] * 0.9 + 0.1


SCENARIOS: dict[str, tuple[str, float, float, int]] = {
    # name: (draw scenario, pass shift, latency shift ms, runs per task)
    "baseline": ("baseline", 0.0, 0.0, RUNS),
    "improved": ("improved", 0.3, -200.0, RUNS),
    "regressed": ("regressed", -0.3, 200.0, RUNS),
    "noisy": ("noisy", 0.0, 0.0, 2),
}


def main() -> None:
    for name, (scenario, shift, latency_shift, runs) in SCENARIOS.items():
        runner = EvaluationRunner(
            SimulatedAgent(scenario, shift, latency_shift),
            [FixtureGrader()],
            RunnerConfig(num_runs=runs, max_concurrency=1),
            decision_spec=DecisionSpec(prompts=PromptSpec(prompt_version=f"prompt-{scenario}")),
        )
        batch = asyncio.run(runner.run(tasks()))
        (HERE / f"{name}.trials.json").write_text(json.dumps(batch.to_dict()) + "\n")
        print(f"{name}: pass_rate={batch.pass_rate:.3f}")


if __name__ == "__main__":
    main()
