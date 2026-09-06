"""Compare two versions of an agent (model / prompt change) with statistics.

This is the canonical "did v2 actually get better, or is it noise?" workflow:

1. Run the SAME eval set against two configurations (here: two prompt versions).
2. Stamp each run with a `DecisionSpec` so the result is attributable to a
   specific model + prompt fingerprint (the runs' provenance records it).
3. Use `compare_runs` -- the paired task bootstrap behind `tracelens compare`
   -- to decide whether the difference is real, practically meaningful, or
   just noise, with the task as the sampling unit.

No API keys: the "agent" is simulated so the example is runnable and
reproducible (seeded). Swap `make_agent(...)` for your real adapter and the
comparison machinery below is identical. Save both runs with
`tracelens run --save-trials` and the same decision is one command:

    tracelens compare v1-trials.json v2-trials.json --metric mean_score --threshold 0.05

Run:  python examples/version_compare.py
"""

import asyncio
import random

from tracelens import (
    CodeGrader,
    DecisionSpec,
    EvalSet,
    EvaluationRunner,
    ModelConfig,
    PromptSpec,
    RunnerConfig,
    SimpleAdapter,
    Task,
    Transcript,
    TrialBatch,
    compare_runs,
)

random.seed(7)  # reproducible distributions across trials

TASKS = EvalSet(
    name="support-replies",
    tasks=[
        Task(task_id=f"ticket-{i}", name=f"ticket-{i}", input_data={"ticket": f"issue {i}"})
        for i in range(6)
    ],
)


def make_agent(quality_mean: float):
    """A stand-in agent whose reply quality is drawn around `quality_mean`."""

    async def agent(input_data: dict) -> dict:
        quality = max(0.0, min(1.0, random.gauss(quality_mean, 0.12)))
        return {"reply": f"Re: {input_data['ticket']}", "quality": round(quality, 3)}

    return agent


class ReplyQualityGrader(CodeGrader):
    def compute_metrics(self, transcript: Transcript, task: Task) -> dict[str, float]:
        return {"quality": float(transcript.final_output["quality"])}

    def determine_pass(self, metrics: dict[str, float], task: Task) -> tuple[bool, float]:
        return metrics["quality"] >= 0.7, metrics["quality"]


def spec_for(version: str, prompt_text: str) -> DecisionSpec:
    return DecisionSpec(
        model=ModelConfig(provider="openai", model_id="gpt-4o-mini", temperature=0.7),
        prompts=PromptSpec.from_prompts(system_prompt=prompt_text, prompt_version=version),
    )


async def run_version(quality_mean: float, spec: DecisionSpec) -> TrialBatch:
    runner = EvaluationRunner(
        SimpleAdapter(make_agent(quality_mean)),
        [ReplyQualityGrader("reply_quality")],
        RunnerConfig(num_runs=10, max_concurrency=1),  # concurrency=1 keeps the seed reproducible
        decision_spec=spec,
    )
    return await runner.run(TASKS)


async def main() -> None:
    v1_spec = spec_for("v1", "Reply to the support ticket.")
    v2_spec = spec_for(
        "v2", "Reply concisely, cite the relevant policy, and propose concrete next steps."
    )

    b1 = await run_version(0.66, v1_spec)  # v1: weaker prompt
    b2 = await run_version(0.82, v2_spec)  # v2: improved prompt

    print("version comparison")
    print("------------------")
    for label, batch, spec in [("v1", b1, v1_spec), ("v2", b2, v2_spec)]:
        scores = [o.score for trial in batch.trials for o in trial.outcomes]
        print(
            f"  {label} [{spec.fingerprint_short}]  "
            f"pass_rate={batch.pass_rate:.0%}  mean_quality={sum(scores) / len(scores):.3f}  "
            f"n={len(scores)}"
        )

    # The task is the sampling unit: each task's mean quality under v1 and v2
    # is paired, and the interval comes from resampling tasks, so the six
    # tickets' different difficulties cancel instead of looking like noise.
    result = compare_runs(
        b1, b2, metric="mean_score", threshold=0.05, seed=0,
        baseline_label="v1", candidate_label="v2",
    )
    print()
    print("\n".join(result.summary_lines()))


if __name__ == "__main__":
    asyncio.run(main())
