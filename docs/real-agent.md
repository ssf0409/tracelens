# Evaluating a Real Agent

You have run [`hello_world.py`](getting-started.md) and watched `add 2+2`
pass nine times. That toy proves the four-piece skeleton — task, adapter,
grader, runner — but it dodges every hard part of evaluating something real:
your agent already runs as a service, its output is structured rather than a
single number, and "correct" is part deterministic shape and part subjective
quality.

This page is the missing middle rung. We evaluate a realistic agent end to
end: an HTTP/JSON service graded by a deterministic gate **and** an LLM judge,
run multiple times so we can talk about capability versus reliability, with
a transcript we can actually read. When you finish, [wiring production
CI](ci-cd-integration.md) is mostly configuration.

Every code block below is lifted from a runnable example in the repository;
each section links to the full file. The public API used here is what you can
import directly from `tracelens` (see [Concepts](concepts.md) for the model
vocabulary).

---

## 1. The scenario

Real agents are rarely a Python function you can call in-process. More often
your agent is *already a service* — it speaks HTTP and returns JSON. That
single fact changes the eval: you point TraceLens at a URL instead of importing
a callable, and you grade a structured payload instead of a scalar.

We use exactly that shape. The full runnable version
([`examples/http_agent_eval.py`](https://github.com/ssf0409/tracelens/blob/main/examples/http_agent_eval.py))
stands up a 20-line stdlib HTTP server as the "agent under test." Swap its URL
for your real service and the rest of the script is unchanged — that is the
point.

---

## 2. Defining the task set

Tasks are the unit of evaluation. Load them from JSON with `JSONTaskLoader`
and wrap them in an `EvalSet`, exactly as
[`examples/run_eval.py`](https://github.com/ssf0409/tracelens/blob/main/examples/run_eval.py)
does:

```python
from pathlib import Path
from tracelens import EvalSet
from tracelens.core.task import JSONTaskLoader

loader = JSONTaskLoader()
tasks = loader.load(Path("examples/tasks.json"))
eval_set = EvalSet(name="Math Examples", tasks=tasks)
```

The JSON shape ([`examples/tasks.json`](https://github.com/ssf0409/tracelens/blob/main/examples/tasks.json))
is a `{"tasks": [...]}` envelope. Each task carries an `input_data` payload
sent to the agent, plus optional `metadata`, `tags`, `category`, and
`difficulty` for filtering and sampling:

```json
{
  "task_id": "math-add-large",
  "name": "Large number addition",
  "input_data": {"a": 1234, "b": 5678},
  "category": "arithmetic",
  "tags": ["addition", "medium"],
  "difficulty": "medium",
  "metadata": {"expected": 6912}
}
```

`metadata.expected` is a hint your deterministic grader reads — TraceLens
itself does not interpret it. You can also build `Task` objects inline, which
is what the HTTP example does:

```python
from tracelens import Task

eval_set = EvalSet(name="http-echo", tasks=[
    Task(name="add small", input_data={"a": 2, "b": 3}),
    Task(name="add large", input_data={"a": 1000, "b": 1234}),
])
```

---

## 3. The adapter: wiring the real agent in

For an HTTP agent, use `HTTPAPIAdapter`. It is configured with an
`HTTPAdapterConfig` — the constructor takes a config object, not loose
keyword arguments:

```python
from tracelens import HTTPAPIAdapter, HTTPAdapterConfig

adapter = HTTPAPIAdapter(
    HTTPAdapterConfig(base_url=base_url, endpoint="/", method="POST"),
)
```

`HTTPAdapterConfig` also accepts `timeout`, `auth` (an `AuthConfig` for bearer
tokens or API keys), `retry` (a `RetryConfig` with exponential backoff), and
`extra_headers`. Keep secrets in environment variables and pass them into
`AuthConfig` — never hard-code them:

```python
import os
from tracelens import AuthConfig, AuthScheme, HTTPAdapterConfig

config = HTTPAdapterConfig(
    base_url="https://api.example.com",
    endpoint="/v1/agent/run",
    auth=AuthConfig(scheme=AuthScheme.BEARER, token=os.environ["AGENT_TOKEN"]),
)
```

By default the adapter sends `task.input_data` as the JSON body and treats the
parsed JSON response as `transcript.final_output`. To reshape either side,
subclass and override `build_request_body(task)` or `parse_response_body(data)`.
The adapter records each HTTP attempt as a tool-call step on the transcript,
which you will read in section 6.

> If your agent is *not* an HTTP service, subclass `AgentAdapter` directly and
> implement `async def run(self, task) -> Transcript`, or wrap a plain async
> function with `SimpleAdapter(fn)` as
> [`examples/run_eval.py`](https://github.com/ssf0409/tracelens/blob/main/examples/run_eval.py)
> does. The runner only depends on the `AgentAdapter` interface, so everything
> downstream is identical.

---

## 4. Grading: a deterministic gate plus an LLM judge

Real grading has two halves. A **deterministic gate** answers "is the output
even well-formed?" and an **LLM judge** scores subjective quality. Combine them
with `CompositeGrader`.

TraceLens expresses how a grader affects CI through `EvalPolicy`, a three-way
enum: `GATE` (any violation fails the trial outright), `WARN` (recorded,
configurably non-blocking), and `TRACK` (pure signal). The older `GraderRole`
vocabulary still works for back-compat (`MUST_PASS` → `GATE`,
`SCORE_CONTRIBUTOR` → `TRACK`), but `EvalPolicy` is the current API — prefer it.

**The gate.** `JsonSchemaGrader` defaults to `GATE`, so a malformed payload
fails the trial regardless of how the judge scores it. Note `schema` is a
keyword-only argument:

```python
from tracelens import JsonSchemaGrader

schema_gate = JsonSchemaGrader(
    "output_schema",
    schema={
        "type": "object",
        "properties": {"answer": {"type": "integer"}, "ok": {"type": "boolean"}},
        "required": ["answer", "ok"],
    },
)
```

**The judge.** `LLMGrader` is an ABC — you implement `build_grading_prompt`
and `parse_llm_response`, and you supply an `LLMProvider` that makes the actual
model call. TraceLens deliberately ships only the `LLMProvider` ABC plus an
`InMemoryProvider` for testing; the vendor call is a small subclass you own.
[`examples/llm_provider_examples.py`](https://github.com/ssf0409/tracelens/blob/main/examples/llm_provider_examples.py)
shows OpenAI and Anthropic providers and an `instruction_following` grader:

```python
import json
from tracelens import LLMGrader, Task, Transcript

class InstructionFollowingGrader(LLMGrader):
    def build_grading_prompt(self, transcript: Transcript, task: Task) -> str:
        return f"Score instruction_following 1-10 for: {transcript.final_output}"

    def parse_llm_response(self, response, task):
        data = json.loads(response)
        score = float(data["instruction_following"]) / 10.0
        return score >= 0.7, score, {"instruction_following": score}, data.get("feedback", "")
```

`parse_llm_response` returns `(passed, score, metrics, feedback)`. The base
class honors `GraderConfig`: each attempt is bounded by `timeout_seconds`, and
malformed responses are retried up to `max_retries` times with exponential
backoff when `retry_on_error` is set.

**Combining them.** `CompositeGrader` takes `(grader, weight)` pairs and
aggregates policy-aware: the weighted average becomes the score, but only
`GATE` failures fail the trial.

```python
from tracelens import CompositeGrader

quality_judge = InstructionFollowingGrader("instruction_following", provider=my_provider)
composite = CompositeGrader(
    grader_id="agent_quality",
    graders=[(schema_gate, 1.0), (quality_judge, 1.0)],
)
```

If your output rules are declarative, `BehaviorContract.to_graders()` generates
the matching graders for you, each carrying its own policy — see
[`examples/contract_eval.py`](https://github.com/ssf0409/tracelens/blob/main/examples/contract_eval.py).

---

## 5. Running it

`EvaluationRunner` drives trials with concurrency control and per-trial
timeouts. Set `num_runs > 1` so you have enough samples to measure
non-determinism:

```python
from tracelens import EvaluationRunner, RunnerConfig

config = RunnerConfig(num_runs=5, max_concurrency=5, timeout_seconds=30.0)
runner = EvaluationRunner(adapter, [composite], config)
batch = await runner.run(eval_set)

print(f"{batch.total_count} trials, pass rate {batch.pass_rate:.1%}")
```

`runner.run` is async — call it from `asyncio.run(main())`. `RunnerConfig`
also exposes `progress_callback`, `checkpoint_path`, and `checkpoint_interval`
for long runs (the CLI surfaces these as `--progress` and `--checkpoint`),
plus `max_infra_retries` to re-attempt `INFRA_ERROR` trials with exponential
backoff. Checkpoint resume skips completed trials, re-runs infra-errored
ones and skipped placeholders, and refuses (with `CheckpointError`) a
checkpoint from a mismatched eval set, adapter, graders, or `DecisionSpec`.
Identity uses class paths, so two configs of the same adapter class are only
told apart when the runner carries a `DecisionSpec`; checkpointing also
requires stable, explicit `task_id`s (auto-generated ids change every run).

The returned `TrialBatch` separates harness failures from agent failures:
alongside `pass_rate` it carries `infra_error_rate` and `grader_error_rate`.
A spike in either means the eval broke, not the agent — watch them.

What counts as an infra error is set by `RunnerConfig.infra_exception_types`
(default `DEFAULT_INFRA_EXCEPTION_TYPES` — `InfraError`, `MemoryError`,
`ConnectionError` — both importable from `tracelens`; the CLI flag is
`--infra-exceptions`). httpx transport errors are not `ConnectionError`
subclasses, so for an HTTP agent either extend the set (e.g.
`infra_exception_types=DEFAULT_INFRA_EXCEPTION_TYPES + (httpx.TransportError,)`)
or raise `InfraError` from a subclassed adapter when a failure is
environmental. A `TimeoutError` raised inside the adapter also classifies
through this set (`FAILED` by default); only the runner's own budget timeout
becomes `TIMEOUT`.

To render reports, hand the batch to `ReportGenerator`:

```python
from tracelens import ReportGenerator

gen = ReportGenerator(k_values=[1, 3, 5], consistency_k_values=[2, 3, 5])
report = gen.build_report(batch)
print(gen.render_ci_summary(report))   # also: render_markdown, render_html
```

---

## 6. Reading a transcript

The whole point of a *trace* lens is that you can read why a trial passed or
failed. Each `Trial` holds a `Transcript`, and a `Transcript` is a list of
`TranscriptStep`s plus convenience accessors:

```python
for trial in batch.trials:
    t = trial.transcript
    print(trial.task_id, "->", t.final_output)
    print("  steps:", t.tool_calls_count, "tool calls,", t.llm_calls_count, "llm calls")
    print("  duration_ms:", t.duration_ms, "tokens:", t.total_tokens)
    if t.has_errors:
        print("  errors:", t.errors)
```

`TranscriptStep` carries a `step_type` (`tool_call`, `llm_call`,
`agent_output`, `error`, ...), an optional `tool_call` record, and `error`
text. The HTTP adapter logs every request attempt as a `tool_call` step named
`http_request`, so when a trial fails you can see the status code and how many
retries it took. Use `t.get_steps_by_type(...)` or
`t.get_tool_calls_by_name("http_request")` to drill in. Reading a handful of
transcripts is the single most reliable way to catch grader bugs and false
signals.

---

## 7. Capability vs reliability

With `num_runs=5` you can ask two different questions. **pass@k** ("can it do
this at all, given k tries?") is a capability ceiling; **pass^k** ("does it
succeed every time across k tries?") is a reliability floor. Both take the
per-task pass results the batch already groups for you:

```python
from tracelens import pass_at_k, pass_to_k

results_by_task = batch.get_pass_results_by_task()  # {task_id: [True, False, ...]}

for task_id, results in results_by_task.items():
    n, c = len(results), sum(results)
    capability = pass_at_k(n=n, c=c, k=3)         # >=1 of 3 attempts passes
    reliability = pass_to_k(results=results, k=3)  # all 3 in a window pass
    print(f"{task_id}: pass@3={capability:.2f}  pass^3={reliability:.2f}")
```

`pass_at_k(n, c, k)` is the unbiased estimator over `n` samples with `c`
passes; `pass_to_k(results, k)` is the fraction of length-`k` windows that are
all-pass. A high pass@k with a low pass^k is the classic "capable but flaky"
signature. `ReportGenerator` already computes both for the `k_values` and
`consistency_k_values` you pass it. For the full intuition and when each one
should gate a release, read
[pass@k vs pass^k](pass-at-k-vs-pass-hat-k.md).

---

## 8. Storing a baseline and wiring CI

Once a run looks good, freeze it as a baseline so future runs are compared
against it instead of graded in a vacuum. `BaselineManager` stores and promotes
baselines (CANARY / CAPABILITY / EXPERIMENTAL types), and `RegressionDetector`
flags declines by severity (NONE / MINOR / MODERATE / SEVERE).

Rather than re-derive that here, follow the dedicated walkthroughs — they use
the real manager and detector APIs end to end:

- [Baseline regression tutorial](baseline-regression-tutorial.md) — store,
  promote, and compare baselines; understand severity levels.
- [CI/CD integration](ci-cd-integration.md) — run the eval in CI and block,
  warn, or route to manual review on regression.

From the CLI the same flow is one command (verify flags against
`tracelens --help`):

```bash
tracelens run \
  --eval-set examples/tasks.json \
  --adapter myproject.adapters.MyAgentAdapter \
  --graders myproject.graders.QualityGrader \
  --num-runs 5 \
  --baseline-check \
  --baselines-file eval/baselines.json \
  --fail-on-regression moderate \
  --save-trials reports/trials.json
```

---

## Where to go next

- [Core Concepts & Glossary](concepts.md) — the full model vocabulary (Task,
  Trial, Transcript, Outcome, DecisionSpec).
- [pass@k vs pass^k](pass-at-k-vs-pass-hat-k.md) — capability versus
  reliability in depth.
- [Baseline regression tutorial](baseline-regression-tutorial.md) and
  [CI/CD integration](ci-cd-integration.md) — gate releases on regressions.
- Runnable sources:
  [http_agent_eval.py](https://github.com/ssf0409/tracelens/blob/main/examples/http_agent_eval.py),
  [llm_provider_examples.py](https://github.com/ssf0409/tracelens/blob/main/examples/llm_provider_examples.py),
  [contract_eval.py](https://github.com/ssf0409/tracelens/blob/main/examples/contract_eval.py),
  [run_eval.py](https://github.com/ssf0409/tracelens/blob/main/examples/run_eval.py).
