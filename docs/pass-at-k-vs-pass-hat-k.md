# pass@k vs pass^k

Two agents can have the *same* average pass rate and yet one is safe to ship
while the other is not. TraceLens reports two metrics that pull apart this
difference:

- **pass@k** — *capability*. "Can the agent do this **at all**, given k tries?"
- **pass^k** — *reliability*. "Will the agent do this **every time**, across k tries?"

They move in opposite directions as `k` grows. Reading the wrong one into a CI
gate is how a flaky agent passes review.

---

## The one-line definitions

| Metric | Question | Formula (per attempt rate `p`) | Direction as `k` ↑ |
|--------|----------|--------------------------------|--------------------|
| `pass@k` | At least **one** success in `k` attempts? | `1 - (1 - p)^k` | **rises** toward 1 |
| `pass^k` | **All** `k` attempts succeed? | `p^k` | **falls** toward 0 |

More attempts can only help "succeed once" and can only hurt "succeed every
time." That is the whole intuition.

> TraceLens estimates both from observed trials rather than a fixed `p`.
> `pass_at_k(n, c, k)` uses the unbiased Chen et al. estimator over `n` trials
> with `c` passes; `pass_to_k(results, k)` is the proportion of length-`k`
> sliding windows in which every trial passed. Windows follow `run_index`
> order, never the order trials happened to finish, and a window never spans
> a missing run (`None` in the sequence).

---

## Truth table: one window, two verdicts

Take any window of `k = 3` trials. `pass@3` asks *any* pass; `pass^3` asks
*all* pass. Only the all-pass window satisfies reliability:

| 3-trial window | `pass@3` counts it? (any pass) | `pass^3` counts it? (all pass) |
|----------------|:------------------------------:|:------------------------------:|
| `P P P` | ✅ | ✅ |
| `P P F` | ✅ | ❌ |
| `P F P` | ✅ | ❌ |
| `F P P` | ✅ | ❌ |
| `F F F` | ❌ | ❌ |

Three of these five windows look fine to a capability metric and fail a
reliability metric. That gap is exactly the flakiness CI needs to catch.

---

## Worked example: same data, diverging metrics

Observed trial sequence over 10 runs (8 passes, 2 failures, an 80% pass rate):

```
P P F P P P F P P P
```

Running TraceLens over this sequence at several `k`:

| k | `pass@k` (capability) | `pass^k` (reliability) |
|---|:---------------------:|:----------------------:|
| 1 | 0.800 | 0.800 |
| 2 | 0.978 | 0.556 |
| 3 | 1.000 | 0.250 |
| 5 | 1.000 | 0.000 |

Same 80% agent. Ask it to succeed **once in 5** and it is effectively certain
(`pass@5 = 1.000`). Ask it to succeed **5 times in a row** and it essentially
never does (`pass^5 = 0.000`). **pass@k rises while pass^k falls** — read only
the first and you ship an agent that fails 1-in-5 of the time.

Reproduce it:

```python
from tracelens.statistics.pass_at_k import pass_at_k
from tracelens.statistics.consistency import pass_to_k

seq = [True, True, False, True, True, True, False, True, True, True]
n, c = len(seq), sum(seq)

for k in (1, 2, 3, 5):
    print(f"k={k}  pass@k={pass_at_k(n, c, k):.3f}  pass^k={pass_to_k(seq, k):.3f}")
```

---

## Which metric goes where

| Context | Use | Why |
|---------|-----|-----|
| **PR / CI gate** | `pass^k` | A merge should require the agent to behave *every* time, not occasionally. Gate on reliability so flakiness blocks the build. |
| **Dashboards / trend tracking** | **both** | Capability and reliability tell different stories over time. A rising `pass@k` with a falling `pass^k` means the agent is learning the task but getting *less* consistent — a regression a single number hides. |
| **Exploratory capability checks** | `pass@k` | When you only want to know whether a model *can* solve a task at all (prompt iteration, model selection, "is this even feasible"), give it `k` tries and measure capability. |

Rule of thumb: **pick `pass@k` while you are still asking "is this possible?"
and switch to `pass^k` once you are asking "can I depend on it?"**

---

## See also

- [`examples/hello_world.py`](https://github.com/ssf0409/tracelens/blob/main/examples/hello_world.py)
  computes both metrics end to end.
- The checked-in
  [sample report](https://github.com/ssf0409/tracelens/blob/main/examples/reports/hello_world_report.md)
  shows pass@k and pass^k side by side for a real run.
- [Evaluation Levels](./evaluation-levels.md) — where these metrics sit in
  function vs task vs system-level evaluation.
- [Accuracy Best Practices](./accuracy.md) — sample sizes and bootstrap CIs so
  these numbers are trustworthy, not noise.
