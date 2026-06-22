# Grader Library

A grader is how TraceLens turns a `Transcript` into a pass/fail decision and a
`0-1` score. You can always write your own by subclassing `CodeGrader` or
`LLMGrader` (see [Concepts](concepts.md)), but most common checks already ship
with the framework. This page catalogs the built-ins so you stop hand-rolling
what you can import.

Every grader listed here is importable from the package root:

```python
from tracelens import JsonSchemaGrader, LatencyGrader, EventChainVerifier
```

Combine several with a `CompositeGrader` — the weighted average becomes the
score, but only `GATE` failures fail the trial. The deterministic-gate +
LLM-judge pattern is worked through in
[Evaluating a Real Agent §4](real-agent.md).

## EvalPolicy: GATE / WARN / TRACK

Each built-in carries a default `EvalPolicy` that decides how its result affects
CI. The three tiers are:

- **GATE** — a violation fails the trial outright. Use for hard constraints
  (valid JSON, no forbidden tool calls).
- **WARN** — recorded and surfaced, but does not block by default. Use for
  budgets and soft regressions.
- **TRACK** — a pure signal that never fails CI. Use for quality tracking.

Override any default by passing a `GraderConfig(policy=...)`:

```python
from tracelens import ContainsGrader, EvalPolicy, GraderConfig

grader = ContainsGrader(
    "disclaimer",
    required=["AI-generated"],
    config=GraderConfig(policy=EvalPolicy.GATE),  # promote TRACK -> GATE
)
```

---

## Output-shape & content graders

These deterministic graders validate the structure and content of
`transcript.final_output`. All of their value arguments are **keyword-only**
(after the positional `grader_id`).

### JsonSchemaGrader

Validates `transcript.final_output` against a JSON Schema using the
`jsonschema` library. An invalid schema raises `ValueError` at construction.

- Signature: `JsonSchemaGrader(grader_id, *, schema, config=None)`
- Default policy: **GATE**

```python
from tracelens import JsonSchemaGrader

grader = JsonSchemaGrader(
    "output_schema",
    schema={
        "type": "object",
        "properties": {"answer": {"type": "integer"}, "ok": {"type": "boolean"}},
        "required": ["answer", "ok"],
    },
)
```

### StructuredOutputGrader

Parses `transcript.final_output` with a Pydantic model loaded at grading time
from a dotted import path. Passes when `model.model_validate(output)` succeeds.

- Signature: `StructuredOutputGrader(grader_id, *, model_path, config=None)`
- Default policy: **GATE**

```python
from tracelens import StructuredOutputGrader

grader = StructuredOutputGrader(
    "typed_output",
    model_path="myproject.models.ResponseSchema",
)
```

### ContainsGrader

Checks that `str(transcript.final_output)` contains every required substring and
none of the forbidden ones.

- Signature: `ContainsGrader(grader_id, *, required, forbidden=None, config=None)`
- Default policy: **TRACK**

```python
from tracelens import ContainsGrader

grader = ContainsGrader(
    "safety_disclaimer",
    required=["reviewed by a human"],
    forbidden=["guaranteed returns"],
)
```

### RegexMatchGrader

Checks that `str(transcript.final_output)` matches every regex pattern via
`re.search`. An invalid pattern raises `ValueError` at construction.

- Signature: `RegexMatchGrader(grader_id, *, patterns, config=None)`
- Default policy: **TRACK**

```python
from tracelens import RegexMatchGrader

grader = RegexMatchGrader(
    "has_ticket_id",
    patterns=[r"TICKET-\d+"],
)
```

### ConstraintGrader

Evaluates a list of heterogeneous constraints. Supported `type` values:
`must_include`, `must_not_include` (substring checks on `str(output)`),
`numeric_range` (`output[field]` within `[min, max]`), and `enum`
(`output[field]` in `values`). An unknown `type` raises `ValueError` at
construction.

- Signature: `ConstraintGrader(grader_id, *, constraints, config=None)`
- Default policy: **GATE**

```python
from tracelens import ConstraintGrader

grader = ConstraintGrader(
    "bounds",
    constraints=[
        {"type": "must_include", "value": "summary"},
        {"type": "numeric_range", "field": "confidence", "min": 0.0, "max": 1.0},
        {"type": "enum", "field": "status", "values": ["ok", "error"]},
    ],
)
```

---

## Budget & performance graders

These read resource and behavior signals off the transcript. Their arguments are
positional (note: not keyword-only, unlike the validators above).

### LatencyGrader

Passes when `transcript.duration_ms <= max_ms`. Score is
`max(0, 1 - duration/max)`. A non-positive `max_ms` raises `ValueError`.

- Signature: `LatencyGrader(grader_id, max_ms, config=None)`
- Default policy: **WARN**

```python
from tracelens import LatencyGrader

grader = LatencyGrader("latency_budget", max_ms=2000.0)
```

### TokenBudgetGrader

Passes when `transcript.total_tokens <= max_tokens`. Score is
`max(0, 1 - total/max)`. A non-positive `max_tokens` raises `ValueError`.

- Signature: `TokenBudgetGrader(grader_id, max_tokens, config=None)`
- Default policy: **WARN**

```python
from tracelens import TokenBudgetGrader

grader = TokenBudgetGrader("token_budget", max_tokens=5000)
```

### ToolCallGrader

Validates tool-call compliance: every tool in `required_tools` must be called,
only tools in `allowed_tools` may be called (when given), and no tool in
`forbidden_tools` may be called.

- Signature:
  `ToolCallGrader(grader_id, required_tools=None, allowed_tools=None, forbidden_tools=None, config=None)`
- Default policy: **GATE**

```python
from tracelens import ToolCallGrader

grader = ToolCallGrader(
    "tool_policy",
    required_tools=["search"],
    forbidden_tools=["delete_account"],
)
```

### TraceConsistencyGrader

Checks agent self-consistency: `tool_error_rate`, `unused_tool_results` (tool
results with no subsequent agent output), and `phantom_calls` (tools called that
are not in `expected_tools`). Passes when `tool_error_rate < 0.5` and there are
no phantom calls.

- Signature: `TraceConsistencyGrader(grader_id, expected_tools=None, config=None)`
- Default policy: **WARN**

```python
from tracelens import TraceConsistencyGrader

grader = TraceConsistencyGrader(
    "trace_health",
    expected_tools=["search", "summarize"],
)
```

---

## Event-chain verification

`EventChainVerifier` answers "did the agent do X, then Y?" — it scans transcript
steps for an ordered (or partially-ordered) sequence of expected events. Unlike
the graders above, it takes a config object, not loose arguments.

- Signature: `EventChainVerifier(grader_id, chain_config, **kwargs)`
- Default policy: inherits the `GraderConfig` default (**TRACK**), since no
  policy is forced in `__init__`. Pass `config=GraderConfig(policy=...)` via
  `kwargs` to change it.

The config is built from three pieces:

- **`EventExpectation`** — one expected event: an `event_id`, a `match_type`,
  the field that match type needs, and an optional `after` list of `event_id`s
  it must follow (used by `PARTIAL` ordering).
- **`EventMatchType`** — how a step is matched: `TOOL_NAME`,
  `TOOL_NAME_AND_ARGS`, `CONTENT_REGEX`, `STEP_TYPE`, or `RESULT_REGEX`.
- **`OrderingMode`** — `STRICT` (events appear in exact order), `UNORDERED`
  (all appear, any order), or `PARTIAL` (respects each event's `after` DAG).

```python
from tracelens import (
    EventChainConfig,
    EventChainVerifier,
    EventExpectation,
    EventMatchType,
    OrderingMode,
)

config = EventChainConfig(
    expected_events=[
        EventExpectation(
            event_id="search",
            match_type=EventMatchType.TOOL_NAME,
            tool_name="search",
        ),
        EventExpectation(
            event_id="analyze",
            match_type=EventMatchType.TOOL_NAME,
            tool_name="analyze",
            after=["search"],  # must come after the search event
        ),
    ],
    ordering=OrderingMode.PARTIAL,
)

grader = EventChainVerifier("did_search_then_analyze", config)
```

`EventChainConfig` also takes `require_all` (default `True` — every event must
match to pass) and `score_per_event` (default `True`). When `require_all` is
`False`, the grader passes once the found-ratio reaches the config's
`pass_threshold`.

---

## Which grader do I want?

| Need | Grader |
| --- | --- |
| Output must match a JSON Schema | `JsonSchemaGrader` |
| Output must parse as a Pydantic model | `StructuredOutputGrader` |
| Output must contain / avoid certain strings | `ContainsGrader` |
| Output must match regex patterns | `RegexMatchGrader` |
| Mixed include / range / enum rules | `ConstraintGrader` |
| Stay under a time budget | `LatencyGrader` |
| Stay under a token budget | `TokenBudgetGrader` |
| Enforce required / allowed / forbidden tools | `ToolCallGrader` |
| Detect flaky tool usage / phantom calls | `TraceConsistencyGrader` |
| Verify an ordered sequence of actions | `EventChainVerifier` |

For declarative output rules, `BehaviorContract.to_graders()` generates a
matching grader per section automatically — see
[`examples/contract_eval.py`](https://github.com/ssf0409/tracelens/blob/main/examples/contract_eval.py).

## See also

- [API Reference](reference.md) — full signatures for every public symbol.
- [Evaluating a Real Agent](real-agent.md) — combining a gate and a judge with
  `CompositeGrader` (§4).
- [Core Concepts & Glossary](concepts.md) — Transcript, Outcome, and the model
  vocabulary these graders read.
- Runnable sources:
  [contract_eval.py](https://github.com/ssf0409/tracelens/blob/main/examples/contract_eval.py),
  [http_agent_eval.py](https://github.com/ssf0409/tracelens/blob/main/examples/http_agent_eval.py).
