"""Compare two versions of an agent (model / prompt change) with statistics.

This is the canonical "did v2 actually get better, or is it noise?" workflow:

1. Run the SAME eval set against two configurations (here: two prompt versions).
2. Stamp each run with a `DecisionSpec` so the result is attributable to a
   specific model + prompt fingerprint.
3. Use `compare_metrics` (bootstrap CI + effect size + permutation p-value) to
   decide whether the difference is statistically real, not run-to-run jitter.

No API keys: the "agent" is simulated so the example is runnable and
reproducible (seeded). Swap `make_agent(...)` for your real adapter and the
comparison machinery below is identical.

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
)
from tracelens.statistics.inference import compare_metrics

random.seed(7)  # reproducible distributions across trials

TASKS = EvalSet(
    name="support-replies",
    tasks=[Task(name=f"ticket-{i}", input_data={"ticket": f"issue {i}"}) for i in range(6)],
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


async def run_version(quality_mean: float, spec: DecisionSpec) -> tuple[object, list[float]]:
    runner = EvaluationRunner(
        SimpleAdapter(make_agent(quality_mean)),
        [ReplyQualityGrader("reply_quality")],
        RunnerConfig(num_runs=10, max_concurrency=1),  # concurrency=1 keeps the seed reproducible
        decision_spec=spec,
    )
    batch = await runner.run(TASKS)
    scores = [o.score for trial in batch.trials for o in trial.outcomes]
    return batch, scores


async def main() -> None:
    v1_spec = spec_for("v1", "Reply to the support ticket.")
    v2_spec = spec_for(
        "v2", "Reply concisely, cite the relevant policy, and propose concrete next steps."
    )

    b1, s1 = await run_version(0.66, v1_spec)  # v1: weaker prompt
    b2, s2 = await run_version(0.82, v2_spec)  # v2: improved prompt

    print("version comparison")
    print("------------------")
    for label, batch, scores, spec in [("v1", b1, s1, v1_spec), ("v2", b2, s2, v2_spec)]:
        mean_q = sum(scores) / len(scores)
        print(
            f"  {label} [{spec.fingerprint_short}]  "
            f"pass_rate={batch.pass_rate:.0%}  mean_quality={mean_q:.3f}  n={len(scores)}"
        )

    res = compare_metrics(s1, s2, confidence=0.95, compute_p_value=True)
    if res.is_significant and res.delta > 0:
        verdict = "v2 is significantly BETTER"
    elif res.is_significant and res.delta < 0:
        verdict = "v2 is significantly WORSE"
    else:
        verdict = "no significant difference (within noise)"

    print()
    print(
        f"  quality delta (v2 - v1) = {res.delta:+.3f}  "
        f"95% CI [{res.ci_lower:+.3f}, {res.ci_upper:+.3f}]  "
        f"cohens_d={res.cohens_d:.2f}  p={res.p_value:.3f}"
    )
    print(f"  -> {verdict}")
    print(f"  fingerprints differ: {v1_spec.fingerprint != v2_spec.fingerprint}")


if __name__ == "__main__":
    asyncio.run(main())
