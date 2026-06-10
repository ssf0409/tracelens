# Evaluation Recipes

TraceLens evaluates; it does not own your domain truth. The cleanest way to use
it for a non-trivial system is a **recipe**: a thin, repeatable arrangement of
the primitives TraceLens already provides, with the domain-specific parts owned
by the projects that have the domain knowledge.

This pattern keeps TraceLens small. Most "can TraceLens do X?" questions are
really "how do I arrange a recipe for X?" — and the answer is usually existing
primitives plus a downstream grader, not a new TraceLens feature.

## The three roles

Separate *who computes the truth*, *who scores it*, and *who acts on it*:

| Role | Owns | Examples |
|------|------|----------|
| **Producer** | The canonical, deterministic evidence to be judged. TraceLens reads it but never recomputes it. | A backtest result, a recorded transcript, a search-result set, a frozen request/response fixture. |
| **Evaluator (TraceLens)** | Loading evidence as tasks, running graders, comparing to a baseline, confidence intervals, and the report. | `Task`, `CodeGrader`, `RegressionDetector`, `BaselineManager`, `ReportGenerator`. |
| **Consumer** | Deciding what to do with the report. | A CI gate, a promotion decision, an operator dashboard. |

The boundary matters: if TraceLens starts recomputing domain truth (simulating
trades, re-running searches), it stops being a general evaluator and becomes a
fork of the producer. Keep evidence as *input*.

## Recipe shape

A recipe wires up these pieces. Only two of them ever live in TraceLens:

| Piece | Owner | Purpose |
|-------|-------|---------|
| Evidence schema | Producer repo | The canonical facts TraceLens may score but not recompute. |
| Task mapping | Recipe / downstream | Wrap each evidence case into `Task.input_data`. |
| Deterministic grader | Recipe / downstream | Compute metrics, pass/fail, score, and reason codes. |
| Threshold config | Downstream repo | Promotion / regression thresholds for that project. |
| Report schema | **TraceLens** | Stable JSON fields for metrics, confidence, decision, reasons. |
| Rollout policy | Consumer repo | How to act on the decision. |

Prefer a documented recipe plus a stable report schema over a first-class
TraceLens subcommand. Add a packaged command only after **two** downstream
projects need the same evaluator — until then, a downstream repo can wrap the
generic CLI with its own friendlier command.

## Generic CLI

A recipe runs on the existing CLI; the domain logic lives behind the adapter
and grader paths:

```bash
tracelens run \
  --eval-set data/eval_set.json \
  --adapter downstream_eval.adapters.EvidenceReplayAdapter \
  --graders downstream_eval.graders.MyDomainGrader \
  --num-runs 1 \
  --output reports/eval_candidate_v2.json
```

The adapter replays producer evidence into a `Transcript`; the grader reads
`Task.input_data`, computes domain metrics, and returns an `Outcome`. TraceLens
handles the rest.

## A three-way decision, not just pass/fail

For promotion gates, a binary pass/fail is often too blunt. A useful recipe maps
its grader output to three decisions and lets the consumer act on each:

- **`reject`** — the candidate is worse than baseline beyond the configured
  thresholds. Block it.
- **`review_first`** — directionally promising but with trade-offs that need a
  human. The default whenever evidence is mixed or the sample is small.
- **`auto_promote`** — broadly better, statistically defensible, with enough
  samples. Only here is unattended promotion appropriate.

Encode these as `reason_codes` in `Outcome.metrics`/`feedback` so the consumer
can act on them without parsing prose. Never `auto_promote` from a tiny sample —
default to `review_first`.

## Worked examples

- **Trading strategy gates** — a backtester produces canonical run evidence,
  deterministic graders score risk/return metrics against the baseline, and the
  deploy pipeline consumes the verdict before promoting a strategy.
- **Retrieval** — compare a candidate chunking policy against a baseline using
  canonical search-result evidence and a deterministic relevance grader.
- **Prompt routing** — compare candidate routing rules against a baseline using
  frozen request fixtures and outcome labels.

In every case the producer owns domain truth, TraceLens owns evaluation and
reporting, and the consumer owns the rollout decision.
