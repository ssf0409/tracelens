# Human-Eval Calibration

LLM-as-judge graders drift. A prompt that scored transcripts well last month
can quietly start over- or under-grading as your agent, your data, or the
judge model changes. The only way to know is to compare the grader against
human judgement on a sample — and to do it again, periodically.

TraceLens covers the whole loop with three steps:

```
run --save-trials   →   sample   →   (human grades)   →   reconcile
   raw trials          worksheet      filled worksheet     agreement report
```

TraceLens deliberately does **not** ship a rating UI or a human-grade store.
You bring the human grades (a spreadsheet, a form, a notebook — anything that
produces JSON); TraceLens owns the selection and the agreement math.

## 1. Keep the raw trials

Calibration grades *transcripts*, so you need the raw trials, not just the
summary report. Pass `--save-trials`:

```bash
tracelens run \
  --eval-set tasks.json \
  --adapter my.Adapter \
  --graders my.Grader \
  --num-runs 5 \
  --output results.json \
  --save-trials trials.json
```

`trials.json` is a serialized `TrialBatch`: every trial with its transcript and
grader outcome. `results.json` is the per-task report `reconcile` reads to get
grader scores (see step 4).

## 2. Select a sample to hand-grade

Nobody hand-grades 1000 transcripts. `sample` picks a small, useful subset and
writes a fill-in worksheet:

```bash
tracelens sample \
  --trials trials.json \
  --size 20 \
  --strategy diverse \
  --output review.json
```

### Choosing a strategy

The strategy is the whole point — a random handful tells you little, but a
well-chosen sample shows you exactly where the grader and a human disagree.

| Strategy | Picks | Use it when |
|----------|-------|-------------|
| `diverse` *(default)* | Trials spread evenly across the score range, including the lowest and highest. | General calibration — you want to see whether the grader tracks humans across the *whole* range, not just the middle. |
| `boundary` | Trials whose score is closest to the 0.5 pass/fail line. | You care most about the pass/fail decision. This is where graders and humans disagree most. |
| `failures` | Failing trials only, lowest score first. | Auditing false negatives — is the grader failing things a human would pass? |
| `random` | A reproducible random sample (set `--seed`). | You want an unbiased estimate of overall agreement. |

`--size` is a maximum; if the batch has fewer gradeable trials, you get all of
them. Trials without a transcript or grader score are skipped automatically.

### The worksheet

`review.json` is a list of rows. The reviewer fills in `human_score` (0–1) and
`human_passed`; the `grader_*` and `output_excerpt` fields are read-only context
to grade against:

```json
[
  {
    "task_id": "add-7-8",
    "human_score": null,
    "human_passed": null,
    "notes": "",
    "grader_score": 0.52,
    "grader_passed": true,
    "output_excerpt": "15"
  }
]
```

## 3. Have a human grade it

Fill `human_score` and `human_passed` for each row. Grade against the *task
expectation*, not against the grader's score — the whole point is to catch the
grader being wrong. A few rules of thumb:

- Score on the same 0–1 scale the grader uses.
- Don't peek at `grader_score` before forming your own judgement.
- Use `notes` to record *why* you disagreed — that's the most actionable output.

The filled file is, by design, a valid TraceLens annotations file. The extra
`grader_*` context columns are ignored on load.

## 4. Reconcile grader vs human

```bash
tracelens reconcile \
  --grader my.Grader \
  --samples tasks.json \
  --results results.json \
  --annotations review.json \
  --threshold 0.7
```

`reconcile` is an alias for `calibrate` — same command, clearer name for this
step. It matches each human annotation to the grader outcome for that task and
reports:

```
Calibration Report
==================================================
Samples:              20
Pearson r:            0.8123
Spearman rho:         0.7950
Pass/Fail agreement:  0.9000
Cohen's kappa:        0.7600
Mean score delta:     0.0450
MAE:                  0.0820
Grader bias:          0.0450
--------------------------------------------------
Threshold:            0.7
Calibrated:           YES
```

- **Pearson / Spearman** — does the grader's score track the human's? `< 0.7`
  means the grader's ranking is unreliable.
- **Pass/fail agreement & Cohen's kappa** — does the grader make the same
  pass/fail call? Kappa corrects for chance agreement.
- **Grader bias** — positive means the grader is systematically generous;
  negative means it's harsh. A large bias with high correlation is fixable by
  adjusting the threshold; low correlation means the prompt needs work.

The command **exits non-zero when Pearson r is below `--threshold`**, so you can
run it in CI (e.g. a weekly scheduled job) and get alerted when a grader drifts.

### Where grader scores come from

`reconcile` needs the grader's outcome per task. Two ways to provide it:

- `--results results.json` — reuse the report from your `run` (fastest).
- `--transcripts transcripts.json` — re-grade transcripts on the fly with
  `--grader`. Use this when you've changed the grader and want to re-check it
  against the *same* human grades.

## When to recalibrate

Calibrate when any input to the judge changes, and on a slow cadence otherwise:

- the judge model is upgraded or swapped;
- the grading prompt changes;
- the agent's output distribution shifts (new domain, new capabilities);
- otherwise, a periodic (e.g. weekly or per-release) 20-sample check.

If correlation drops below `0.7`, tune the grading prompt — the `notes` from
disagreements are your best guide — and re-run `reconcile` against the same
annotations to confirm the fix. See [accuracy.md](accuracy.md) for sample-size
and prompt-tuning guidance.
