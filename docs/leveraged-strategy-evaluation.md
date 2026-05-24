# TraceLens Design Suggestion: Leveraged Strategy Evaluation Gates

Date: 2026-05-24
Target project: TraceLens
Producer project: `crypto-trading-system`
Consumer project: `agent-os`
Status: design suggestion for review

## Summary

TraceLens should evaluate whether leverage-aware strategy recommendations are
consistently better than existing recommendations across historical cases. It
should not compute liquidation prices or simulate margin. That truth belongs in
`crypto-trading-system`. TraceLens should consume canonical leveraged survival
backtest results and provide repeatable scoring, regression detection, and
promotion gates.

## Goals

1. Define an evaluation task for leverage-aware trading recommendations.
2. Compare candidate recommendation policies against baseline policies.
3. Detect regressions where a new policy improves one case but worsens overall
   survival, liquidation, or risk-adjusted return.
4. Provide promotion criteria that can be run locally and in CI.
5. Produce reviewable reports that explain why a policy passed or failed.

## Non-Goals

- Do not implement exchange liquidation math in TraceLens.
- Do not fetch live market data from exchanges.
- Do not decide live trading actions.
- Do not replace `crypto-trading-system` backtesting or risk models.
- Do not promote policies from tiny sample sizes.

## Roles And Responsibilities

TraceLens owns:

- dataset/task representation for leveraged recommendation evaluation
- grader definitions
- baseline comparison
- confidence intervals and regression detection
- promotion report formatting

`crypto-trading-system` owns:

- canonical leveraged survival backtest outputs
- liquidation, fee, slippage, and margin assumptions
- symbol-specific price-path simulation
- paper account snapshots and simulated paper-order decisions

`agent-os` owns:

- deciding when to request evaluation
- consuming pass/fail and report metadata
- using evaluation status as a review-first or auto-promotion gate

## Evaluation Input

Each evaluation case should include:

```json
{
  "case_id": "BTC-USD-long-governance_v1-2026-05-01",
  "symbol": "BTC-USD",
  "side": "long",
  "strategy": "governance_v1",
  "market_regime": "high_volatility",
  "baseline_recommendation": {
    "max_allowed_leverage": 20,
    "stop_loss_pct": 2.0,
    "take_profit_pct": 4.0,
    "position_size_pct": 2.0
  },
  "candidate_recommendation": {
    "max_allowed_leverage": 10,
    "stop_loss_pct": 1.5,
    "take_profit_pct": 3.0,
    "position_size_pct": 2.0
  },
  "leveraged_survival_result": {
    "schema_version": "leveraged_survival_backtest.v1",
    "margin_mode": "isolated",
    "leverage_results": [],
    "paper_order_summary": {}
  }
}
```

The `leveraged_survival_result` field should be treated as source evidence,
not recomputed.

## Evaluation Outputs

TraceLens should produce:

```json
{
  "task_id": "leveraged_strategy_eval",
  "candidate_policy": "risk_cap_v2",
  "baseline_policy": "risk_cap_v1",
  "sample_size": 180,
  "passed": false,
  "score": 0.62,
  "metrics": {
    "liquidation_rate_delta": -0.08,
    "survival_rate_delta": 0.09,
    "expected_return_after_fees_delta": -0.01,
    "bad_promotion_rate": 0.04,
    "over_block_rate": 0.12,
    "under_block_rate": 0.02
  },
  "confidence": {
    "method": "bootstrap_ci",
    "survival_rate_delta_ci": [0.03, 0.14],
    "expected_return_delta_ci": [-0.04, 0.01]
  },
  "decision": "review_first",
  "reason_codes": [
    "survival_improved",
    "return_delta_not_positive",
    "over_block_rate_above_auto_promotion_threshold"
  ]
}
```

## Core Metrics

| Metric | Meaning |
|---|---|
| `liquidation_rate_delta` | Candidate minus baseline liquidation rate. Lower is better. |
| `survival_rate_delta` | Candidate minus baseline survival rate. Higher is better. |
| `expected_return_after_fees_delta` | Candidate minus baseline return after fees. Higher is better. |
| `bad_promotion_rate` | Candidate allows a leverage tier that baseline blocks and the tier later liquidates. Lower is better. |
| `over_block_rate` | Candidate blocks a tier that baseline allows and the tier survives profitably. Lower is better. |
| `under_block_rate` | Candidate allows a tier that should have been blocked by survival evidence. Lower is better. |
| `paper_order_reject_delta` | Candidate minus baseline simulated paper-order rejection rate. Lower is better when rejection is caused by policy mistakes; neutral when rejection avoids liquidation. |
| `coverage_rate` | Fraction of cases with enough canonical backtest evidence. Higher is better. |

## Promotion Semantics

TraceLens should return one of three decisions:

- `auto_promote`: candidate improves survival without unacceptable return or
  opportunity-cost regression, with enough sample size.
- `review_first`: candidate is directionally promising but has trade-offs that
  need human review.
- `reject`: candidate worsens liquidation risk, survival, or expected return
  beyond configured thresholds.

Suggested MVP thresholds:

| Gate | Suggested Default |
|---|---:|
| Minimum cases | 50 |
| Minimum symbols | 3 |
| Max bad promotion rate for auto-promote | 1% |
| Max over-block rate for auto-promote | 10% |
| Required survival-rate delta for auto-promote | > 0 with CI lower bound >= 0 |
| Required liquidation-rate delta | < 0 with CI upper bound <= 0 |
| Expected return delta for auto-promote | >= -1% absolute |

The MVP should default to `review_first` when evidence is mixed. Auto-promotion
is appropriate only when the improvement is broad, statistically defensible,
and does not hide major opportunity cost.

## Expectations For Integration

- The first version should run from a local fixture dataset.
- TraceLens should accept canonical survival backtest results as input, not call
  crypto-TS services directly.
- TraceLens may score simulated paper-order decisions emitted by crypto-TS, but
  it should not model account state or order operations itself.
- Reports should be JSON first, with human-readable summaries as a secondary
  convenience.
- Baseline and candidate policy versions should be explicit in every report.
- Promotion decisions should include reason codes that downstream systems can
  display without parsing prose.
- Small samples should return `review_first` or `reject`, never `auto_promote`.

## Dataset Expectations

The dataset should be versioned and deterministic:

- immutable case IDs
- canonical backtest result blobs or stable references
- policy versions for baseline and candidate
- market regime labels when available
- no live network dependency during evaluation
- reproducible local run from a clean checkout

This allows `agent-os` to answer the original decision-quality question: are
we making better decisions as more data arrives, or are we only fitting recent
cases?

## Local CLI Shape

A downstream-friendly command might look like:

```bash
tracelens leveraged-strategy-eval \
  --dataset data/leveraged_strategy_cases.jsonl \
  --baseline-policy risk_cap_v1 \
  --candidate-policy risk_cap_v2 \
  --output reports/leveraged_strategy_eval_risk_cap_v2.json
```

The exact command name can change, but the behavior should stay stable:
load deterministic cases, run graders, compare against baseline, write JSON.

## Success Criteria

1. TraceLens can evaluate a fixture dataset without network or exchange access.
2. Evaluation uses canonical `crypto-trading-system` leveraged survival results
   as input rather than recomputing liquidation math.
3. A candidate that reduces liquidation by blocking every trade is not
   automatically promoted if over-block/opportunity cost is too high.
4. A candidate that improves one symbol but regresses the aggregate dataset is
   rejected or marked `review_first`.
5. A candidate can only be `auto_promote` when sample size, confidence interval,
   liquidation delta, survival delta, and opportunity-cost gates pass together.
6. Reports include reason codes that `agent-os` can surface in its own decision
   journal and PR review comments.

## Integration With Agent-OS

`agent-os` should use TraceLens as the promotion/evaluation layer:

1. Ask `crypto-trading-system` for canonical leveraged survival backtests.
2. Build or refresh a deterministic evaluation dataset.
3. Ask TraceLens to compare candidate policy vs baseline.
4. Treat `reject` as IIEF `wait`.
5. Treat `review_first` as IIEF `measure`.
6. Treat `auto_promote` as eligible for IIEF `commit`, subject to normal
   operator and risk gates.

This keeps the boundaries clean: crypto-TS computes trading truth, TraceLens
evaluates policy improvement, and agent-os chooses the next action.

## Open Questions For TraceLens Maintainers

1. Should this be a generic evaluation recipe using existing TraceLens
   primitives, or a first-class domain evaluator?
2. Should promotion thresholds live in TraceLens config, agent-os config, or
   the evaluation dataset metadata?
3. Should TraceLens store baseline histories itself, or should downstream
   repos own the persisted baseline artifacts and call TraceLens as a library?

The recommended MVP is a generic evaluation recipe plus JSON report schema.
Make it first-class only after two downstream projects need the same evaluator.
