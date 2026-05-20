"""The smallest possible tracelens run — three minutes from `pip install` to output.

What this shows, end to end:

1. **A task** — what the agent must do, plus the expected answer.
2. **An agent** — any async callable. Wrapped via ``SimpleAdapter`` so tracelens
   can drive it.
3. **A grader** — deterministic code that scores the agent's output.
4. **A run** — one task, one trial, pass/fail in milliseconds.

There is no LLM call, no network, no config file. Once you understand
this script, the other examples (``contract_eval.py``, ``http_agent_eval.py``,
``noise_aware_regression.py``) just add layers on the same skeleton.

Run it::

    python examples/hello_world.py
"""

import asyncio

from tracelens import (
    CodeGrader,
    EvalSet,
    EvaluationRunner,
    RunnerConfig,
    SimpleAdapter,
    Task,
    Transcript,
)


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
        config=RunnerConfig(num_runs=1),
    )
    batch = await runner.run(eval_set)

    print("\ntracelens hello-world")
    print("--------------------")
    print(f"trials run : {len(batch.trials)}")
    print(f"pass rate  : {batch.pass_rate:.0%}")
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
