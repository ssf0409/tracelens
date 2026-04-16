# high-stakes-autonomous benchmark pack

Flagship eval-kit benchmark that **demonstrates infra-noise awareness end-to-end**.
Reproduces Anthropic's "Quantifying infrastructure noise in agentic coding
evals" (Feb 2026) finding at miniature scale: the same agent on the same
tasks can swing from 100 % pass rate to 50 % pass rate purely because of
resource-budget changes — and eval-kit correctly attributes that swing to
infrastructure rather than capability.

## What it exercises

| eval-kit feature                                        | Proven by                                                                                                   |
| ------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `DecisionSpec.infra` in fingerprint                     | Baseline and current runs have different `fingerprint_short` values when only `memory_hard_limit_mb` differs |
| `TrialStatus.INFRA_ERROR` classification                 | OOM-killed trials show up as `INFRA_ERROR`, never inflating the "model failed" rate                         |
| `TrialBatch.infra_error_rate` + `ReportGenerator`        | `Infra-Error Rate: 50.0%` shows up in markdown, CI line, and HTML summary cards                              |
| `RegressionDetector.compare_with_specs()` noise-aware   | Report includes the `⚠ Infra config mismatch detected` warning                                              |
| `SafetyGateGrader` with `EvalPolicy.GATE`               | Must-pass safety constraints are checked independently of capability graders                                 |

## Layout

```
benchmarks/high-stakes-autonomous/
├── README.md             # This file
├── tasks.json            # 6 tasks: 2 safety (GATE), 4 capability
├── agent.py              # BudgetedAgent — memory-budget-sensitive mock
├── graders.py            # SafetyGateGrader, CapabilityGrader
└── run_benchmark.py      # Runner + reporter + baseline comparison
```

`baseline_generous.json` is generated locally (gitignored); see the walkthrough below.

## Walkthrough

Each command runs in about one second — the mock agent is deterministic and
designed purely to exercise the framework, not to test LLM capability.

### 1. Record a baseline under generous resources

```bash
python benchmarks/high-stakes-autonomous/run_benchmark.py \
  --memory-hard-limit-mb 2048 \
  --num-runs 3 \
  --save-baseline benchmarks/high-stakes-autonomous/baseline_generous.json
```

Expected: **18/18 pass** (6 tasks × 3 runs), fingerprint `edf0f6897c85…`
(your exact hash will differ — `InfraConfig.hostname` is observational
and excluded from the fingerprint, but any other spec field in the
environment will change it).

### 2. Re-run under a tight memory cap

```bash
python benchmarks/high-stakes-autonomous/run_benchmark.py \
  --memory-hard-limit-mb 512 \
  --num-runs 3 \
  --compare-baseline benchmarks/high-stakes-autonomous/baseline_generous.json
```

Expected:

- **Pass Rate: 50.0%** — only the 3 tasks within budget (2 safety + 1 lean
  capability) pass.
- **Infra-Error Rate: 50.0%** — the 3 heavy capability tasks all OOM and
  are classified as `INFRA_ERROR`, not `FAILED`. The runner's status
  breakdown labels each trial:
  ```
  [INFRA_ERROR ] capability-heavy-data-stack  run=0 — container OOM: needed 1536MB, hard limit 512MB
  ```
- **Fingerprint changes** — `edf0f6897c85` → something different because
  `memory_hard_limit_mb` is in the hash.
- **Regression comparison** — per-task diff with severity tags, followed by:
  ```
  ⚠ Infra config mismatch detected between baseline and current run.
    Regressions within the 3pp noise band have been flagged but NOT
    counted as blocking — rerun under the baseline's resource config
    to confirm whether the capability really dropped.
  ```

### 3. Rerun under the original config to confirm the regression is noise

```bash
python benchmarks/high-stakes-autonomous/run_benchmark.py \
  --memory-hard-limit-mb 2048 \
  --num-runs 3 \
  --compare-baseline benchmarks/high-stakes-autonomous/baseline_generous.json
```

Expected: 18/18 pass, no regressions, no infra-error rate. Demonstrates
that matching the baseline's resource config eliminates the apparent
capability drop — exactly the Anthropic remediation.

## Adding your own tasks

The pack is intentionally small. To extend it, edit `tasks.json` and add
entries with:

- `category: "safety"` + `expected_behavior: "refuse"` or
  `"confirm_before_irreversible"` to exercise the `SafetyGateGrader`.
- `category: "capability"` + `memory_mb_needed: <N>` to exercise the
  resource-sensitivity path.

The mock agent is in `agent.py` — swap `BudgetedAgent` for a real
`HTTPAPIAdapter` or `SimpleAdapter(my_agent_fn)` in `run_benchmark.py`
to evaluate a real system under the same infrastructure-noise framework.

## Background

- Anthropic, "Quantifying infrastructure noise in agentic coding evals"
  (Feb 2026): https://www.anthropic.com/engineering/infrastructure-noise
- Anthropic, "Demystifying evals for AI agents":
  https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents
