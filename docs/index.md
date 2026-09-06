# TraceLens

**Evaluation and regression-testing for AI agents.** TraceLens turns agent runs
into inspectable traces, graded outcomes, baseline comparisons, calibration
data, and CI-ready reliability signals.

It is deliberately domain-agnostic: TraceLens evaluates evidence, while your
project owns the task data, agent invocation, graders, and rollout policy. And
it is deliberately local: tasks, baselines, run artifacts, and decisions are
files in your repository, nothing needs an account or a server, and CI runs
the same command you run at your desk.

---

## Start here

<div class="grid cards" markdown>

- :material-rocket-launch: **[Getting Started](getting-started.md)**

    From a fresh checkout to your first eval run in five minutes — no LLM keys,
    no config files.

- :material-cube-outline: **[Core Concepts & Glossary](concepts.md)**

    The evaluation pipeline and every object — Task, Transcript, Outcome, Trial
    — explained on one page.

- :material-school: **[pass@k vs pass^k](pass-at-k-vs-pass-hat-k.md)**

    Capability vs reliability, with a truth table and which metric belongs in a
    CI gate.

- :material-cog: **[CI/CD Integration](ci-cd-integration.md)**

    Wire TraceLens into GitHub Actions with regression gating. Copy-paste
    workflow included.

- :material-book-open-variant: **[User Guide](user-guide.md)**

    Every public API explained, with decision trees for graders, adapters, and
    analysis methods.

</div>

---

## What TraceLens gives you

New to the vocabulary below? The [Core Concepts & Glossary](concepts.md) page
defines every object and shows how they connect.

- **Inspectable traces** — every run becomes a `Transcript` you can read.
- **Outcome grading** — deterministic `CodeGrader`, LLM-as-judge `LLMGrader`,
  and `CompositeGrader`, plus declarative `BehaviorContract`s.
- **Statistical rigor** — `pass@k` for capability, `pass^k` for reliability,
  and bootstrap confidence intervals so signals aren't noise.
- **Baseline regression detection** — canary, capability, and experimental
  baselines with severity-graded CI gates.
- **Harness-vs-agent separation** — `infra_error` and `grader_error` rates are
  tracked separately, so a broken eval never looks like a failing agent.
- **Decisions with their evidence** — `tracelens run --baseline-check` records
  one gate decision in every output; `tracelens compare` gives a verdict between
  two saved runs with an interval and a practical threshold; `tracelens inspect`
  explains a failure from the trials file.
- **Provenance, not promises** — every run records what was measured (task
  content hashes, graders, settings) and which candidate was under test.
  Comparisons refuse mismatched measurements, and "what changed" is reported
  as attribution evidence, not proof of cause.

### What is demonstrated today

| Claim | What it rests on |
|---|---|
| The documented workflow works from a fresh install | A CI job installs a freshly built wheel into a clean environment and drives `init`, `run --config`, baselines, the gate, an intentional regression, `inspect`, `compare`, a targeted rerun, an infra outage, a grader crash, malformed input, and checkpoint/resume through the console script. |
| The statistics do what the contract says | Hand-derived and independent-reference tests against the statistical contract (task-level bootstrap with multiplicity, order-independent pass^k, paired run comparison). |
| Integrations behave at their boundaries | Tests exercise the JSON/JSONL/CSV loaders, the HTTP adapter, the optional Hugging Face loader, the generated GitHub workflow, and `tracelens.yaml`. |
| TraceLens caught a real regression in a real project | **Not yet published.** The examples and the scaffold use simulated agents. A sanitized downstream case study is the open half of issue #33; until it exists, treat "catches regressions" as a tested mechanism, not an observed result. |

```bash
pip install tracelens
```

See [Installation](installation.md) for `uv`, extras (`[llm]`, `[http]`), and
development setup.

---

## How the docs are organized

- **Get Started** — install, run hello-world, and the example ladder.
- **Concepts** — evaluation levels, the two reliability metrics, and accuracy
  best practices.
- **Guides** — end-to-end walkthroughs for baselines, human calibration, and CI.
- **Reference** — the auto-generated [API Reference](reference.md).
- **Contributing** — testing tiers and the release process.
