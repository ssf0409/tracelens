# Supported Evaluation Scenarios

TraceLens is built for agent evaluation workflows where you need to inspect
what happened, score the outcome, compare it against a baseline, and gate CI
when reliability regresses.

## Good Fits

### Task-Level Agent Evaluation

Use TraceLens when one task maps to one agent run:

- Goal decomposition.
- Code generation or code review.
- Research synthesis.
- Customer-support response generation.
- Planning, routing, or tool-selection decisions.

Start with `Task`, an `AgentAdapter`, one or more graders, and
`EvaluationRunner`.

### Regression Testing For Agents

Use baselines when you have a known-good agent or prompt and want to prevent
silent quality loss:

- Compare pass rate, quality score, safety score, or latency to a baseline.
- Keep canary baselines protected.
- Auto-promote capability baselines when a new version improves enough.
- Fail CI on moderate or severe regressions.

Start with `BaselineManager`, `RegressionDetector`, and
`docs/ci-cd-integration.md`.

### Inspectable Trace Review

Use transcripts when debugging why a run passed or failed:

- Final output.
- Intermediate transcript steps.
- Timing.
- Decision fingerprint.
- Grader feedback.

Start with `Transcript`, `TranscriptStep`, and the report generator.

### Infrastructure-Noise-Aware Evals

Use `DecisionSpec.InfraConfig` when resource differences can change the result:

- Memory limits.
- CPU limits.
- Timeout budgets.
- Concurrency.
- Harness or sandbox provider.

TraceLens can mark small regressions as within the infrastructure noise band
when the agent config is unchanged but infra changed.

Start with `examples/noise_aware_regression.py` and
`benchmarks/high-stakes-autonomous/`.

### HTTP Agent Evaluation

Use `HTTPAPIAdapter` when the system under test is already exposed as a service:

- Local FastAPI or Flask apps.
- Hosted agent APIs.
- Internal platform endpoints.
- Agent wrappers that already speak JSON.

Start with `examples/http_agent_eval.py` and install `tracelens[http]`.

### Contract-Based Evaluation

Use behavior contracts when your expected behavior is easy to declare:

- Required output schema.
- Must-include or must-not-include fields.
- Safety constraints.
- Quality warnings.
- Tracking-only metrics.

Start with `examples/contract_eval.py`.

## Less Good Fits

TraceLens is not trying to be:

- A model benchmark leaderboard.
- A full observability backend.
- A prompt management platform.
- A replacement for human review.
- A hosted evaluation service.

You can still export TraceLens outputs into those systems, but the core package
focuses on local, inspectable, CI-ready evaluation.

## Choosing A First Example

| You have... | Start with |
|-------------|------------|
| A Python function or local agent | `examples/hello_world.py` |
| A JSON HTTP endpoint | `examples/http_agent_eval.py` |
| A strict behavioral contract | `examples/contract_eval.py` |
| A baseline and new candidate run | `examples/noise_aware_regression.py` |
| A safety/resource-sensitive workflow | `benchmarks/high-stakes-autonomous/` |

## Minimum Production Setup

For a real project, aim for:

- 20-50 real tasks collected from actual failures or important workflows.
- At least one must-pass grader for safety or format constraints.
- At least one score-contributor grader for quality.
- `num_runs > 1` for non-deterministic agents.
- A canary baseline for critical behavior.
- CI that fails on blocking regressions.
- A habit of reading transcripts when a metric changes.
