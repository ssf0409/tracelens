# TraceLens

**Evaluation and regression-testing for AI agents.** TraceLens turns agent runs
into inspectable traces, graded outcomes, baseline comparisons, calibration
data, and CI-ready reliability signals.

It is deliberately domain-agnostic: TraceLens evaluates evidence, while your
project owns the task data, agent invocation, graders, and rollout policy.

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
