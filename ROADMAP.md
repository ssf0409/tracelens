# TraceLens Roadmap

TraceLens is a thin-core evaluation and regression-testing framework for AI
agents. The core package should stay small, local, inspectable, and CI-ready.
Domain truth, rollout policy, hosted dashboards, and framework-specific agent
integration belong in downstream projects or examples.

## North Star

**From many recorded agent runs, surface the variables that actually move
outcomes — and turn them into rollout, debugging, and prioritization
decisions.**

An eval framework measures outcomes. TraceLens should also explain outcome
variance. Every run already records three families of variables alongside the
outcome:

- **Configuration variables** — `DecisionSpec`: model, prompts, tools, agent,
  and infrastructure, with fingerprints and machine-readable diffs.
- **Task variables** — category, tags, difficulty, and metadata on every
  `Task`.
- **Behavior variables** — steps, tool calls, errors, tokens, and latency on
  every `Transcript`.

The north star is the layer that joins these across many trials and answers,
with statistical honesty, the three questions a team actually has:

1. **Ship it?** Did this change make the agent better or worse — overall and
   on which slices — and was the difference real or noise?
2. **Where do I look?** Which failure modes exist, which variables separate
   passing from failing trials, and which transcripts should a human read
   first?
3. **What do I fix first?** Which failure cluster or task slice, if fixed,
   buys the most expected reliability?

A change moves toward the north star when it makes one of these questions
answerable from saved run data with less manual work than before.

This is an extension of what TraceLens already does, not a pivot. Noise-band
regression comparison, infra-config diffing, harness-vs-agent failure
separation, and human-eval sampling are all early forms of the same idea:
attribute outcome differences to their causes before making a decision.

## Product Doctrine

**Thin core, rich recipes.**

TraceLens should own the reusable evaluation primitives:

- Task, transcript, trial, outcome, and grader models.
- Agent adapters and execution orchestration.
- pass@k, pass^k, bootstrap confidence intervals, and regression detection.
- Baselines, reproducibility fingerprints, and CI summaries.
- Human-eval calibration worksheets and reconciliation.
- Trial analysis: slicing, failure grouping, driver association, and run
  comparison over recorded trial data.

The analysis layer is batch, local, files-in/files-out computation over
recorded trials. It does not change the non-goals below.

TraceLens should not become:

- A hosted tracing or observability backend.
- A prompt playground or prompt-management platform.
- A broad provider SDK wrapper.
- A domain-specific benchmark suite hidden inside the core package.
- A RAG metrics framework unless that need is proven by multiple real users.

When a request could be solved by either a core feature or a recipe, prefer the
recipe until at least two downstream projects need the same abstraction.
Core abstractions and analysis primitives remain disciplined by real downstream
evidence (tracked in issue #33), rather than expanding ahead of proven need.

## The Gap Today & Shipped Foundations

The initial 0.4 series identified key architectural gaps; 0.5.0 closed several
foundational seams (shipped in 0.5.0: YAML run configuration, provenance tracking,
paired run comparison via `tracelens compare`, inspect/offline HTML reporting, and
persisted gate reporting across formats via issues #28, #44-#47):

- **Results slicing (open):** Task tags, categories, and difficulty are
  captured and filterable before a run, but descriptive slicing joining them back
  to outcomes and exploratory slice comparisons (issue #90, #108) remain open.
- **Failures taxonomy (open):** Statuses, error messages, and grader feedback
  are recorded, but formal grouping across error signatures remains in progress.
- **Trace behavior & outcomes association (open):** Transcripts carry
  steps, tool errors, tokens, and latency, but associating features with pass/fail
  remains to be built.
- **Evidence at the source (open):** Externally produced traces require an import
  contract (issue #104), and adapter step recording helpers remain open (#131).
- **Runs comparison (shipped in 0.5.0):** `tracelens compare` provides paired
  task-level comparisons between saved runs (issue #28).
- **Regression reporting (shipped in 0.5.0):** Report generation threads gate results
  cleanly across JSON, Markdown, and HTML without dropping regression data on
  re-render (issue #47). Report building still accepts an unused baseline manager parameter
  for historical compatibility.

## Executable Roadmap

The issue tracker remains the source of truth for current priority, ownership,
and acceptance criteria; the delivery tracker is issue #55. The estimator,
sampling-unit, and trial-validity definitions every statistic follows live in
`docs/statistical-contract.md`. The phases below are direction, ordered by
leverage; each names its definition of done. CLI commands that do not exist
yet are marked *(proposed)*.

### Phase 0 — Make the existing numbers trustworthy (Shipped in 0.5.0)

Repair the seams so the current decision output is correct and flows end to
end.

- Landed statistical correctness fixes against the statistical contract:
  bootstrap multiplicity (issue #44), order-independent pass^k (issue #45),
  and explicit metric availability instead of zeros (issue #46).
- Threaded regression results through report building so markdown, HTML, JSON,
  and CI renderings include them without CLI-side assembly, and stop dropping
  them when re-rendering saved results (issue #47).
- Rendered harness-health signals that are already computed: grader error
  rate and token totals alongside infra error rate.

**Status:** Completed and shipped in TraceLens 0.5.0.

### Phase 1 — See the variables

Build the analysis foundation: one flat, joinable view of trials.

- Transcript capture helpers and an import path for externally recorded
  transcripts with offline grading (issues #104, #131), so behavior variables
  are actually populated and traces can enter without TraceLens driving the agent.
- A trial-analysis table joining trials with task dimensions (category, tags,
  difficulty, metadata), decision-spec fields, extracted behavior features,
  and outcomes. This is the substrate for descriptive slicing.
- Descriptive slice analysis (issue #90): pass rate and score by any task dimension,
  with sample counts and availability guards.
- Failure grouping: group failed trials by status and normalized error
  signature, with counts and representative trial IDs.
- Surface both in reports (a failure-analysis section) and inspection tools.

Done when, from a saved trials file and its eval set, one command answers:
which task slices fail most, with what failure modes, and which transcripts
to read first.

### Phase 2 — Explain the variance

Turn comparison and exploratory slice diagnosis into decision output.

- `tracelens compare` *(issue #28; shipped in 0.5.0 as a paired task bootstrap)*:
  A/B two saved runs with bootstrap significance, noise-band awareness,
  printing the decision-spec diff — "here is what changed" —
  next to "here is what moved". When the diff shows exactly one changed
  factor, say so: the outcome change is attributable to that factor, which
  is evidence, not causal proof.
- Exploratory slice comparisons *(proposed follow-up in issue #108)*: contract-backed
  paired task comparisons on defined subsets with multiplicity correction and
  sample-size guards (explicitly non-causal).
- A diagnosis view combining failure clusters and read-these-first transcripts.

Done when a user deciding whether to ship runs one comparison and gets a
verdict with evidence, and a user debugging a regression gets a clear list
of moving tasks plus specific transcripts to read.

### Phase 3 — Decide from the corpus

Scale from single runs to accumulated history (exploratory, non-committed ambition).

- File-based run storage conventions and cross-run trends over time.
- Production traces as input: an OpenTelemetry ingestion recipe (issue #7) so
  real traffic can feed the same analysis, starting as an example.
- Optional LLM-assisted failure labeling as an extra, never a core
  dependency.

Done when trends over time are inspectable from accumulated runs, and
downstream projects can feed non-eval traces through the same analysis.

### Supporting tracks

These continue in parallel and feed the north star:

- **Fast first evaluation** — scaffolding, CLI affordances, examples, and CI
  templates until a first eval feels routine (issue #35: `run --config`).
- **Data portability** — JSON native; CSV and JSONL in core; hosted datasets
  and framework-specific loaders as optional integrations or recipes.
- **Recipe ecosystem** — LangChain/LangGraph, OpenAI SDK, OpenTelemetry, RAG
  workflows, and benchmark packs as examples first (issue #26); promote into
  core only after multiple downstream users prove the shape.
- **Dogfooding** — a small self-eval suite (issue #33) is the natural testbed
  for every analysis feature above.
- **Positioning** — once Phase 1 ships, propagate the north-star framing to
  the README and docs so the scope is obvious before users install.

## Guardrails

The analysis layer is only worth building if it stays honest:

- **Correlation is not causation.** Driver rankings are suspects with
  evidence, not verdicts. A decision-spec diff with a single changed factor
  supports *attributing* an outcome change to that factor; it is still not
  proof of causation, so reports say what changed and what moved, never
  "caused by".
- **Small samples are the norm.** Every insight carries confidence intervals
  and minimum-sample guards; multi-way slicing labels or corrects for
  multiple comparisons rather than shipping false discoveries.
- **Grade outcomes, not paths.** Behavior features explain outcomes; they do
  not grade them. Path-based grading remains an explicit opt-in via the
  existing budget and event-chain graders.
- **Domain-agnostic core.** Domain-specific feature extractors and failure
  labelers plug in via dotted import paths and live downstream, exactly like
  adapters and graders do today.

## Contributor On-Ramp

Contributors should start with scoped issues that have clear acceptance
criteria and a small review surface. The [open issues list](https://github.com/ssf0409/tracelens/issues?q=is%3Aissue%20is%3Aopen)
and delivery tracker (issue #55) track active work and delivery status; the GitHub `good first issue`
label serves as an optional discovery aid. Priority labels, milestones, and project status live
in GitHub rather than baking them into this file.

Before opening a PR, read [CONTRIBUTING.md](CONTRIBUTING.md), comment on the
issue with your intended approach, and run `make verify`.

## Release Hygiene

Every release should leave these channels agreeing with each other:

- `CHANGELOG.md` has a dated section for the version.
- The git tag exists on GitHub.
- PyPI shows the same latest version.
- GitHub Releases has a release for the tag (the release workflow creates it
  from the changelog section; a missing section fails the release before
  anything is published). The tag itself comes from merging the release
  pull request that the "Release prepare" workflow opens.
- `mkdocs build --strict` exits cleanly with no project link or rendering
  warnings.

If any channel drifts, fix that before starting more feature work. Trust is a
feature.
