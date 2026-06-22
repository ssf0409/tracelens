# Reproducibility & DecisionSpec

Two evaluation runs with the "same" setup can still diverge. A different model
snapshot, a tweaked system prompt, a tool that quietly bumped its schema, or even
a tighter memory budget on the CI runner can all move your pass rate. When that
happens, the question is always the same: *did the agent regress, or did the
environment change underneath it?* If you never recorded the configuration that
produced a result, you can't answer it.

`DecisionSpec` is TraceLens's answer. It records the model, prompts, tools, agent,
infrastructure, and environment that produced a run, and reduces them to a single
content hash — the **fingerprint**. Identical configurations produce identical
fingerprints; any behavior-affecting change produces a different one. Baselines
stamped with a fingerprint become attributable: a regression carries the exact
config that produced it.

```python
from tracelens import DecisionSpec, ModelConfig, PromptSpec, ToolSpec, AgentSpec

spec = DecisionSpec(
    model=ModelConfig(provider="anthropic", model_id="claude-3-opus-20240229", temperature=0.7),
    prompts=PromptSpec.from_prompts(system_prompt="You are a helpful assistant..."),
    tools=[ToolSpec(name="search", version="1.0")],
    agent=AgentSpec(agent_name="goal_decomposition", agent_version="1.0.0"),
)
print(spec.fingerprint)        # full SHA-256 hex digest
print(spec.fingerprint_short)  # first 12 chars, e.g. "a1b2c3d4e5f6"
```

Every sub-config is exported from the package root (`from tracelens import
DecisionSpec, ModelConfig, ...`), and every field below is optional unless the
table marks it **required**.

## The sub-configs

A `DecisionSpec` composes six optional sub-configs plus two top-level fields
(`global_seed: int | None` and `extra: dict[str, Any]`). All are optional — fill
in only what is meaningful for your setup. The more you populate, the more
precisely two runs can be told apart.

### ModelConfig

What model produced the output, and with what decoding parameters.

| Field | Type | Notes |
|---|---|---|
| `provider` | `str` | **Required.** e.g. `"anthropic"`, `"openai"`, `"google"` |
| `model_id` | `str` | **Required.** e.g. `"claude-3-opus-20240229"` |
| `model_version` | `str \| None` | Specific snapshot if available |
| `temperature` | `float \| None` | `0.0` = deterministic |
| `top_p` | `float \| None` | Nucleus sampling |
| `top_k` | `int \| None` | Top-k sampling |
| `max_tokens` | `int \| None` | Max tokens to generate |
| `seed` | `int \| None` | Provider seed, if supported |
| `stop_sequences` | `list[str] \| None` | Sorted before hashing (order-independent) |
| `extra_params` | `dict[str, Any]` | Any additional model-specific params |

### PromptSpec

What prompts the agent ran. By default `PromptSpec` stores **only SHA-256 hashes**
of prompt text, not the prompts themselves — so a fingerprint is traceable
without committing long or sensitive prompts to your baseline files.

| Field | Type | Notes |
|---|---|---|
| `system_prompt_hash` | `str \| None` | SHA-256 of the system prompt |
| `prompt_template_hash` | `str \| None` | SHA-256 of the main template |
| `prompt_version` | `str \| None` | Version identifier for the prompt set |
| `system_prompt` | `str \| None` | Full text (optional, debugging only) |
| `prompt_template` | `str \| None` | Full text (optional, debugging only) |

Use the `from_prompts(...)` classmethod rather than hashing by hand — it computes
the hashes for you:

```python
PromptSpec.from_prompts(
    system_prompt="You are a helpful assistant...",
    prompt_template="Given {context}, do {task}...",
    prompt_version="v3",
    store_full_prompts=False,  # default: store hashes only
)
```

Only `system_prompt_hash`, `prompt_template_hash`, and `prompt_version` enter the
fingerprint. The optional full-text fields are for debugging and never affect the
hash.

### ToolSpec

A single tool available to the agent. Pass a list of these as `DecisionSpec.tools`.
Tools are sorted by `name` before hashing, so declaration order doesn't matter.

| Field | Type | Notes |
|---|---|---|
| `name` | `str` | **Required.** Tool name |
| `version` | `str \| None` | Tool version |
| `description_hash` | `str \| None` | Hash of the tool description (affects LLM tool selection) |
| `schema_hash` | `str \| None` | Hash of the tool input/output schema |

### AgentSpec

Which agent was evaluated, and which version of its wiring.

| Field | Type | Notes |
|---|---|---|
| `agent_name` | `str` | **Required.** Name of the agent |
| `agent_version` | `str \| None` | Agent version |
| `agent_graph_hash` | `str \| None` | Hash of agent graph structure (multi-agent systems) |
| `config_hash` | `str \| None` | Hash of agent configuration |

### InfraConfig

The runtime environment — resource limits, time budgets, concurrency, and
platform. Agentic evals are end-to-end system tests, so infrastructure is a
first-class experimental variable, not passive scaffolding: record a guaranteed
allocation **and** a separate hard kill threshold per resource, and capture the
sandbox provider because enforcement semantics differ.

**Behavior-affecting fields (included in the fingerprint):**

| Field | Type | Notes |
|---|---|---|
| `cpu_guaranteed` | `float \| None` | Guaranteed CPU in whole cores |
| `cpu_hard_limit` | `float \| None` | CPU kill threshold in whole cores |
| `memory_guaranteed_mb` | `int \| None` | Guaranteed memory (MB) |
| `memory_hard_limit_mb` | `int \| None` | OOM threshold (MB) |
| `time_budget_seconds` | `float \| None` | Per-task wall-clock budget |
| `concurrency_level` | `int \| None` | Max concurrent trials |
| `runtime_platform` | `str \| None` | e.g. `"kubernetes"`, `"docker"`, `"local"` |
| `sandbox_provider` | `str \| None` | Sandboxing provider |
| `harness_version` | `str \| None` | Version of the eval harness |

**Observational fields (deliberately excluded from the fingerprint):**

| Field | Type | Notes |
|---|---|---|
| `hostname` | `str \| None` | Host running the eval |
| `container_id` | `str \| None` | Container / pod ID |
| `wall_clock_start_utc` | `datetime \| None` | UTC start time |

The observational fields are excluded so that two runs with identical resource
configs on *different hosts* collide to the same fingerprint — see
[The fingerprint](#the-fingerprint) below.

### EnvironmentSpec

Build and deployment provenance for traceability.

| Field | Type | In fingerprint? |
|---|---|---|
| `git_commit` | `str \| None` | Yes |
| `git_branch` | `str \| None` | **No** |
| `build_id` | `str \| None` | Yes |
| `runner_version` | `str \| None` | Yes |
| `framework_version` | `str \| None` (TraceLens version) | Yes |
| `python_version` | `str \| None` | **No** |

`git_branch` and `python_version` are recorded for auditing but do not enter the
fingerprint.

## The fingerprint

`DecisionSpec.fingerprint` is the full SHA-256 hex digest; `fingerprint_short` is
its first 12 characters. Both are computed deterministically: each sub-config
contributes a hash dict, the combined dict is serialized with `sort_keys=True`,
and that is hashed. The guiding rule:

> **Config goes in. Observations stay out.** Anything that changes *what the agent
> does* belongs in the fingerprint. Anything that merely records *where or when it
> ran* does not — so the same configuration on a different machine yields the same
> fingerprint.

That is why `InfraConfig.hostname`, `container_id`, and `wall_clock_start_utc`,
plus `EnvironmentSpec.git_branch` and `python_version`, are excluded.

**Different config → different fingerprint:**

```python
a = DecisionSpec(prompts=PromptSpec.from_prompts(system_prompt="Be terse."))
b = DecisionSpec(prompts=PromptSpec.from_prompts(system_prompt="Be verbose."))
assert a.fingerprint != b.fingerprint   # the prompt changed
```

**Same config, different host → same fingerprint:**

```python
from tracelens import InfraConfig
cfg = dict(memory_hard_limit_mb=2048, runtime_platform="kubernetes")
run_a = DecisionSpec(infra=InfraConfig(**cfg, hostname="node-7"))
run_b = DecisionSpec(infra=InfraConfig(**cfg, hostname="node-12"))
assert run_a.fingerprint == run_b.fingerprint   # only the hostname differs
```

One compatibility detail: `infra` is only mixed into the hash when it is explicitly
set, so fingerprints recorded before `InfraConfig` existed stay stable.

Two helpers complement the fingerprint: `spec.diff(other)` returns the
field-level differences as `{field: (self_value, other_value)}`, and
`spec.is_compatible_with(other)` returns `True` when two specs share the same
model provider/`model_id` and agent name — useful for comparing a prompt change
while holding the model fixed.

## Stamping a run

You rarely build fingerprints by hand. Pass a `decision_spec` to the
`EvaluationRunner` and it stamps the spec onto every transcript that doesn't
already carry one:

```python
from tracelens import EvaluationRunner, RunnerConfig, SimpleAdapter, DecisionSpec, ModelConfig

runner = EvaluationRunner(
    adapter=SimpleAdapter(my_agent),
    graders=[...],
    config=RunnerConfig(num_runs=5),
    decision_spec=DecisionSpec(
        model=ModelConfig(provider="anthropic", model_id="claude-3-opus-20240229"),
    ),
)
batch = await runner.run(eval_set)
```

If an adapter already attached its own `decision_spec` to a transcript, the runner
leaves it untouched; otherwise it fills in the runner-level spec. The result: every
trial in the batch records the configuration that produced it, so any baseline you
promote from this batch carries its fingerprint.

## Infra-noise-aware regression

Because `InfraConfig` is part of the fingerprint, the regression detector can tell
"the agent/model/prompt changed" apart from "only the infrastructure changed." This
matters: resource-config changes alone can swing agentic-eval scores by several
percentage points — often more than the gap between frontier models.

`RegressionDetector.compare_with_specs(...)` takes both the baseline and current
specs. When their `InfraConfig` differs, it sets `report.infra_config_mismatch`
(with the field-level `report.infra_config_diff`) and marks sub-noise-band
regressions as `within_noise_band=True`, so default CI gates don't block on a
delta that is plausibly just infra noise:

```python
report = RegressionDetector(min_delta_percent=1.0).compare_with_specs(
    baseline,
    current_results=[{"pass_rate": current_rate}] * 6,
    baseline_spec=baseline_spec,   # DecisionSpec(infra=InfraConfig(memory_hard_limit_mb=2048, ...))
    current_spec=current_spec,     # DecisionSpec(infra=InfraConfig(memory_hard_limit_mb=512,  ...))
)

print(report.infra_config_mismatch)            # True — memory budget changed
for key, (was, now) in report.infra_config_diff.items():
    print(f"  {key}: {was!r} -> {now!r}")       # memory_hard_limit_mb: 2048 -> 512
print(len(report.blocking_regressions))        # excludes within_noise_band ones
print(report.should_block_ci())
```

`report.blocking_regressions` filters out anything flagged `within_noise_band`, and
`should_block_ci()` keys off that filtered set. The full runnable example —
including an `InfraError`-raising adapter that simulates an OOM kill so infra
failures are classified separately from task failures — is at
[`examples/noise_aware_regression.py`](https://github.com/ssf0409/tracelens/blob/main/examples/noise_aware_regression.py).

## See also

- [Comparing Versions](comparing-versions.md) — the applied "did my model/prompt
  change behavior?" workflow that builds on these fingerprints.
- [Core Concepts & Glossary](concepts.md) — where `Transcript` and the run pipeline fit.
- [Baseline Regression Tutorial](baseline-regression-tutorial.md) — recording baselines and gating CI.
- [API Reference](reference.md) — full signatures for every public symbol.
