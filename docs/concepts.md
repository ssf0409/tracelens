# Core Concepts & Glossary

Read this once and the rest of the docs click into place. TraceLens has a small
vocabulary of objects that pass an agent run through a fixed pipeline: define
the work, run the agent, record what happened, grade the outcome, and aggregate
across repeats so you can reason about capability *and* reliability.

## The pipeline

Every evaluation is the same left-to-right flow. The toy
[hello-world](getting-started.md) run and a production CI gate differ only in
what plugs into each box — not in the shape.

```
   Task            AgentAdapter         Transcript          Grader
┌──────────┐      ┌────────────┐      ┌────────────┐      ┌──────────┐
│ what the │─────▶│ how to run │─────▶│ what the   │─────▶│ did it   │
│ agent    │      │ the agent  │      │ agent did  │      │ succeed? │
│ must do  │      └────────────┘      │ (the run   │      └────┬─────┘
└──────────┘                          │  record)   │           │
                                      └────────────┘           ▼
                                                          ┌──────────┐
                                                          │ Outcome  │
                                                          │ pass +   │
                                                          │ score +  │
                                                          │ feedback │
                                                          └────┬─────┘
                                                               │  one repeat = one
                                                               ▼  ┌──────────┐
                                                                  │  Trial   │
   EvaluationRunner repeats this N times per task, collecting     └────┬─────┘
   every Trial into a ───────────────────────────────────────────────▶ │
                                                               ┌────────▼────────┐
                                                               │   TrialBatch    │
                                                               │ pass@k, pass^k, │
                                                               │ error rates,    │
                                                               │ baseline check  │
                                                               └────────┬────────┘
                                                                        ▼
                                                                    Report
                                                          (markdown / JSON / HTML / CI)
```

The **four pieces you author** are Task, Adapter, Grader, and the Runner config.
Everything else — Transcript, Outcome, Trial, TrialBatch — TraceLens produces
for you.

## Glossary

| Term | One-line definition | Class(es) |
|------|--------------------|-----------|
| **Task** | A single thing the agent must do, plus its input and expectations. | `Task`, `EvalSet` (a set of tasks) |
| **Adapter** | The glue that knows *how to invoke your agent* and return a Transcript. | `AgentAdapter`, `SimpleAdapter`, `HTTPAPIAdapter` |
| **Transcript** | The record of one run: final output, intermediate steps, timing, fingerprint. | `Transcript`, `TranscriptStep` |
| **Grader** | Turns a Transcript into a pass/score judgement. Deterministic or LLM-as-judge. | `CodeGrader`, `LLMGrader`, `CompositeGrader` |
| **Outcome** | The result of grading one Transcript: passed, score, feedback, and error flags. | `Outcome`, `GradeLevel` |
| **Trial** | One Task run once and graded — a Transcript + its Outcome. | `Trial`, `TrialStatus` |
| **TrialBatch** | All Trials for a run, with aggregate statistics and error rates. | `TrialBatch` |
| **Runner** | Drives the whole pipeline: parallelism, repeats, timeouts, checkpointing. | `EvaluationRunner`, `RunnerConfig` |
| **DecisionSpec** | A reproducibility fingerprint of the exact agent config (model, prompt, tools, infra). | `DecisionSpec` |
| **Baseline** | A stored known-good result a candidate run is compared against to detect regressions. | `BaselineManager`, `RegressionDetector` |

## Two distinctions that matter

**Capability vs. reliability.** A run is summarized by two metrics, not one:
`pass@k` (can it succeed *at all* in k tries?) and `pass^k` (does it succeed
*every* time?). They move in opposite directions, and confusing them is how a
flaky agent passes review. This is important enough to have its own page:
[pass@k vs pass^k](pass-at-k-vs-pass-hat-k.md).

**Harness failure vs. agent failure.** A failing grade is not the same as a
broken eval. TraceLens tracks two separate error rates on every `TrialBatch`:

- `infra_error` — the environment failed the run (network dropped, OOM,
  sandbox died — or any exception type you classify as infra via
  `RunnerConfig.infra_exception_types`). A run that blows its time budget is
  `TIMEOUT` and counts against the agent, not as infra.
- `grader_error` — a grader threw while judging (the *grading harness* broke).

A spike in either means your evaluation is broken, **not** that the agent got
worse — so they are reported separately from the pass rate. Never let a harness
failure masquerade as a regression.

## Where to go next

- [Getting Started](getting-started.md) — see this pipeline run end to end in
  five minutes.
- [Build Your First Eval](quickstart.md) — author your own Task, Grader, and
  Runner.
- [pass@k vs pass^k](pass-at-k-vs-pass-hat-k.md) — the capability/reliability
  split in depth.
- [API Reference](reference.md) — every class above, from its docstrings.
