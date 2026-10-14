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
- [x] #184 RED/GREEN: both directions, both better-direction declarations,
  zero/nonzero limits, exact boundary, relative negative/zero baselines,
  invalid/nonfinite data, persistence/update/promotion/history, CI mean flags.
- [x] Add a frozen task and runnable replay gate. Store baseline once; reload it
  on subsequent runs, preserve it byte-for-byte, persist trials and gate evidence.
- [x] Subprocess E2E: unchanged, increase, decrease, missing metric, new metric,
  no metrics, invalid baseline. Check exit codes and saved findings.
- [x] Run `make verify`, `make docs`, the example directly, and the E2E tests
  against a built wheel in a clean environment outside the editable checkout.
- [x] Review the final diff and prepare scoped commits and PR descriptions.

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
- #184 initial API and recipe checks: `70 passed in 26.76s`; full gate before
  independent review: `1503 passed in 168.99s`, coverage 96.19%.
- Independent review found three important counterexamples despite those checks:
  zero-baseline relative change broke JSON persistence, mutated nonfinite
  tolerances serialized as disabled nulls, and relative limits could overflow.
  Each has a failing regression followed by a fix; `10 passed in 1.94s`.
- Persistence validates configured bounds before opening the baseline file;
  this preserves existing model assignment behavior while preventing an invalid
  bound from becoming disabled on disk. Relative overflow is scaled before the
  comparison. The recipe represents nonfinite derived fields as unavailable
  without changing the legacy API dictionary.
- Final targeted API and recipe validation: `80 passed in 26.61s`.
- Rebuilt wheel installed into `/tmp/tracelens-canary-wheel`; its installed
  manager source matches the checkout byte-for-byte. All installed-package
  journeys: `15 passed in 66.41s`, run outside the editable checkout.
- Direct wheel-backed replay: record/check unchanged exit 0; increase, decrease,
  missing, new, empty, and zero-channel exit 1. All checks leave baseline bytes
  unchanged and persisted `gate.json` agrees with stdout and the process exit.
- Review scope: the statistical CLI gate remains separate, CI tolerance checks
  the mean only, and the example validates an offline workflow. External
  downstream services are not claimed as exercised by this fixture.
- Final full regression gate after review fixes: `1513 passed in 154.48s`,
  coverage 96.20%, `verify passed`; lint and strict typechecking passed.
  Strict docs build: `Documentation built in 1.82 seconds`.
- Deliver two review units: strict coverage first, then two-sided tolerance on
  top of it. Both remain draft until maintainer review and merge.
