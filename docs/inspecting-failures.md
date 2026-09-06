# Debugging a Failed Evaluation

A failed run should be explainable from the files it wrote, without opening
raw JSON. This page is the loop: **see what failed → read why → fix → rerun
just that task → confirm**.

## 1. Keep the trials

The explanation lives in the trials file, not in the summary. The scaffold's
`tracelens.yaml` already saves it (`outputs.trials: eval/results/trials.json`);
with flags, add `--save-trials eval/results/trials.json` to `tracelens run`.

## 2. See what failed, and why

```bash
tracelens inspect eval/results/trials.json --failures --eval-set eval/tasks.json
```

```text
Inspected eval/results/trials.json: 10 trial(s), run 0eff51c4-…, TraceLens 0.5.0
  passed 7, agent failure 2, infra error 1, grader error 0, not run 0
Selected 3 trial(s) (kinds: agent failure, infra error, grader error)

[1] starter-capital run 0  agent failure  status=completed  attempts=1  duration=1 ms
    why:      the agent ran and a grader failed it (a timeout counts)
    task:     Answer a simple geography question
    input:    {"answer": "Paris", "question": "What is the capital of France?"}
    expected: missing (the task declares no expected output)
    actual:   {"answer": "wrong"}
    grader:   starter FAIL score=0.00 feedback: expected 'Paris', got 'wrong'
    transcript: 1 step(s) (1 shown), 0 tokens, 1 llm call(s), 0 tool call(s)
      1. llm_call  content: missing
...
```

The first two lines of each trial answer the two questions that matter:

- **What kind of failure.** `agent failure` means the agent ran and a grader
  failed it (a timeout counts). `infra error` means the harness could not run
  it, and `grader error` means a grader crashed; those two say nothing about
  the agent, so fix the environment or the grader before reading further. The
  header counts every kind across the whole run, so the failures you are
  looking at are always in proportion.
- **Expected versus actual.** With `--eval-set` each trial shows the task's
  name, input, and declared expected output next to what came back. Without
  it the line says so explicitly; nothing is ever left blank.

Then come the grader lines (verdict, score, metrics, feedback) and the
transcript (steps, tokens, tool calls, errors). Anything the trial did not
record reads `missing`, and anything cut for length ends with a count of what
was cut.

## 3. Narrow it down

- `--task-id t-fail` for one task, `--grader quality` for the trials that
  grader failed or crashed on, `--kind infra` for one kind, `--all` to include
  passing trials, `--limit 5` to cap the output (the number of matches is
  still reported).
- `--html eval/results/failures.html` writes a self-contained page that opens
  offline, with collapsible transcripts, and reads on a phone. `--json` writes
  the same view as data for scripts.
- Output is bounded by default: 400 characters per field and 20 steps per
  transcript, with omitted content counted. `--full` removes the bounds;
  full transcripts may contain sensitive content, so share that output
  deliberately.

`inspect` exits 0 whenever the file could be read, failures or not: it
reports, the baseline gate decides. Input errors exit 2.

## 4. Fix, then rerun just that task

```bash
tracelens run --config tracelens.yaml --task-id starter-capital
```

`--task-id` (repeatable; `run.task_ids` in the config) runs only the named
tasks and refuses ids that are not in the eval set. It is a separate run: its
provenance, its reports, and its checkpoint identity cover the selected tasks
only, so do not reuse the full run's `--checkpoint` path, and read its pass
rate as "these tasks", not the suite.

## 5. Confirm on the whole suite

Run the full suite again. The baseline gate (`--baseline-check`) or
`tracelens compare previous-trials.json new-trials.json` is the evidence that
the fix helped and nothing else regressed; a targeted rerun proves only that
the named task now passes.

## See also

- [User Guide](user-guide.md#from-the-cli) — every `run` flag.
- [Comparing Versions](comparing-versions.md) — deciding whether a change
  helped.
- [CI/CD Integration](ci-cd-integration.md#reading-the-gate) — what a blocked
  gate says and how to act on it.
