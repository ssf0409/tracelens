# Deterministic canary delivery plan

Strict coverage makes a changed measurement set visible; two-sided tolerances
make unexpected deterministic output changes visible. The consumer is a Python
gate over a frozen replay. TraceLens remains domain-neutral.

## Scope and decisions

- Deliver #185 before #184. Default comparison dictionaries retain their shape.
- `coverage="strict"` compares the union of metric names. Noncomparable rows
  identify the missing side; missing data is never zero or passing.
- Optional per-metric change tolerances are finite and nonnegative. Any enabled
  limit exceeded in either direction sets `changed`; equality is within tolerance.
  Relative limits use `abs(baseline)`, including a zero baseline.
- The CI method supports coverage and change flags, while significance remains
  independent. A tolerance checks the current mean, not every sample.
- New metric settings survive save/load, updates retaining thresholds, promotion,
  and historical snapshots. Legacy files keep their comparison behavior.
- The CLI statistical gate uses a separate comparison path. This delivery is the
  Python API and a runnable downstream gate recipe; its policy and exit codes are
  explicit in the recipe and are not new `tracelens run` flags.

## Acceptance and progress

- [x] Establish a clean `make verify` baseline on current main.
- [x] #185 RED/GREEN: missing/new metrics, empty input, no task baseline,
  legacy dictionary compatibility, CI coverage and invalid coverage options.
- [ ] #184 RED/GREEN: both directions, both better-direction declarations,
  zero/nonzero limits, exact boundary, relative negative/zero baselines,
  invalid/nonfinite data, persistence/update/promotion/history, CI mean flags.
- [ ] Add a frozen task and runnable replay gate. Store baseline once; reload it
  on subsequent runs, preserve it byte-for-byte, persist trials and gate evidence.
- [ ] Subprocess E2E: unchanged, increase, decrease, missing metric, new metric,
  no metrics, invalid baseline. Check exit codes and saved findings.
- [ ] Run `make verify`, `make docs`, the example directly, and the E2E tests
  against a built wheel in a clean environment outside the editable checkout.
- [ ] Review the final diff and prepare scoped commits and PR descriptions.

## Coordination and evidence

Implementation authorized by the maintainer in this chat on 2026-10-13.
Before starting, issue #184/#185 comments were empty and no open PR title/body
referenced either issue. Main verified at `daf46555fa64c6bbc0aeab0db30cac34b69aaa94`.
Existing #166 overlaps gate semantics; this delivery does not alter that gate.
Verification outputs and any changes to these decisions are recorded below.

- Main: `1433 passed in 134.68s`, coverage 93.50%, `verify passed`.
- #185 targeted API and recipe subprocess checks: `22 passed in 13.89s`.
- #185 complete gate: `1455 passed in 120.30s`, coverage 96.10%, `verify passed`.
  Strict documentation build: `Documentation built in 1.88 seconds`.
- CI coverage also retains an unknown metric with an empty sample list;
  dropping it would silently hide a measurement-set change (RED/GREEN covered).
