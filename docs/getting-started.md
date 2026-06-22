# Getting Started in Five Minutes

This is the fastest path from a fresh checkout to your first eval run.
No LLM keys, no config files, no boilerplate. After this guide you'll
understand the four-piece skeleton every TraceLens run uses, and you'll
know which example to read next when you have a real agent.

---

## 1. Install

For using TraceLens as a dependency:

```bash
# Recommended: uv
uv pip install tracelens

# Or: plain pip
pip install tracelens
```

For local development and to run the repository examples:

```bash
git clone https://github.com/ssf0409/tracelens.git
cd tracelens
uv pip install -e ".[dev]"
```

The `[dev]` extra pulls in pytest, ruff, mypy, and type stubs so
the standard local verification commands run out of the box:
`pytest -q`, `ruff check src/ tests/`, and `mypy src/tracelens/`.

The published package gives you the `tracelens` library and CLI. The
checkout gives you the `examples/` files used in the demo below.

---

## 2. Run hello-world

```bash
python examples/hello_world.py
```

You should see:

```
tracelens hello-world
--------------------
trials run : 9
pass rate  : 100%
report json: examples/reports/hello_world_report.json
sample md  : examples/reports/hello_world_report.md

  add-2-2                  runs=3  pass_rate=100% output='4'
  add-10-5                 runs=3  pass_rate=100% output='15'
  add-7-8                  runs=3  pass_rate=100% output='15'
```

Render the generated report through the CLI:

```bash
tracelens report --results examples/reports/hello_world_report.json --format markdown
```

Then read the checked-in sample:
[`examples/reports/hello_world_report.md`](https://github.com/ssf0409/tracelens/blob/main/examples/reports/hello_world_report.md).
It includes tasks, trials, pass@k, pass^k, graders, baseline comparison,
regression result, and CI summary.

That's the entire framework, end to end. One small file, sub-second
runtime, no external services. Open
[`examples/hello_world.py`](https://github.com/ssf0409/tracelens/blob/main/examples/hello_world.py) in your editor
and read it top to bottom — the comments map every line to the
four-piece skeleton.

---

## 3. The Four-Piece Skeleton

Every TraceLens run combines four pieces:

| Piece | What it answers | Concrete classes |
|-------|-----------------|------------------|
| **Task** | What does the agent need to do? | `Task`, `EvalSet` |
| **Adapter** | How do I invoke the agent? | `SimpleAdapter`, `HTTPAPIAdapter`, custom subclass of `AgentAdapter` |
| **Grader** | Did the agent get it right? | `CodeGrader` (deterministic), `LLMGrader` (judge model), or auto-generated from a `BehaviorContract` |
| **Runner** | Drive the run, parallelise, collect results. | `EvaluationRunner`, `RunnerConfig` |

If you can describe each piece in a sentence for *your* agent, you're
ready to write your own eval.

---

## 4. The Example Ladder

The examples in `examples/` go from trivial to production-grade,
each adding exactly one new concept. Read them in order:

| Step | File | New concept |
|------|------|-------------|
| 1 | [`hello_world.py`](https://github.com/ssf0409/tracelens/blob/main/examples/hello_world.py) | The four-piece skeleton, in 50 lines, no LLM. |
| 2 | [`contract_eval.py`](https://github.com/ssf0409/tracelens/blob/main/examples/contract_eval.py) | `BehaviorContract.to_graders()` — declare the contract once, get a full grader suite for free. |
| 3 | [`http_agent_eval.py`](https://github.com/ssf0409/tracelens/blob/main/examples/http_agent_eval.py) | `HTTPAPIAdapter` for evaluating a remote agent over HTTP, plus `JsonSchemaGrader` for output-shape gating. |
| 4 | [`noise_aware_regression.py`](https://github.com/ssf0409/tracelens/blob/main/examples/noise_aware_regression.py) | `DecisionSpec` fingerprinting, `RegressionDetector`, and the 3 percentage-point infra-noise band — the production CI gate. |
| 5 | [`llm_provider_examples.py`](https://github.com/ssf0409/tracelens/blob/main/examples/llm_provider_examples.py) | LLM-as-judge: wire OpenAI/Anthropic SDKs into `LLMGrader` via `LLMProvider` (runs offline with a fake provider). |
| 6 | [`human_eval_calibration.py`](https://github.com/ssf0409/tracelens/blob/main/examples/human_eval_calibration.py) | Reconcile an automated grader against human grades to detect LLM-judge drift (no API keys). |
| 7 | [`version_compare.py`](https://github.com/ssf0409/tracelens/blob/main/examples/version_compare.py) | Compare two versions (model/prompt) with bootstrap significance + `DecisionSpec` fingerprints (no API keys). |

Each example is self-contained — running it directly gives you working
output. Copy the one that matches your problem and edit from there.

---

## 5. Where to Go Next

Your immediate next step is to write your own eval:

- **[Build Your First Eval](./quickstart.md)** — author your own `Task`, custom
  grader, and CLI workflow (the natural step 2 after this guide).
- **[Core Concepts & Glossary](./concepts.md)** — the full pipeline and every
  object (`Transcript`, `Outcome`, `Trial`, `TrialBatch`) on one page.

When your agent is real and your eval set has grown, move on to:

- **[Evaluating a Real Agent](./real-agent.md)** — the end-to-end walkthrough
  for a realistic HTTP/LLM agent: schema gate + LLM judge, reading transcripts,
  capability vs reliability, and a baseline.
- **[User Guide](./user-guide.md)** — every public API explained, with
  decision trees for choosing graders, adapters, and analysis methods.
- **[Supported Scenarios](./scenarios.md)** — which agent-evaluation
  problems TraceLens fits, and which first example to copy.
- **[Evaluation Levels](./evaluation-levels.md)** — function vs task vs
  system-level evaluation; pass@k vs pass^k semantics.
- **[pass@k vs pass^k](./pass-at-k-vs-pass-hat-k.md)** — capability vs
  reliability, with a truth table and which metric belongs in a CI gate.
- **[Accuracy Best Practices](./accuracy.md)** — how to keep LLM-judge
  graders calibrated to humans (the difference between "we ran an eval"
  and "we trust this eval").
- **[CI/CD Integration](./ci-cd-integration.md)** — wiring TraceLens into
  GitHub Actions with regression gating.
- **[High-Stakes Autonomous Benchmark](https://github.com/ssf0409/tracelens/blob/main/benchmarks/high-stakes-autonomous/README.md)**
  — the flagship benchmark pack that demonstrates TraceLens's
  infra-noise-aware regression detection on safety-critical tasks.

---

## 6. The 60-Second Mental Model

TraceLens is opinionated about two things, and ergonomic about
everything else:

1. **Grade outcomes, not paths.** A `CodeGrader` looks at
   `transcript.final_output`. It doesn't care which tools the agent
   called or in what order — that's an implementation detail.
2. **Reproducibility is a first-class config.** Every run carries a
   `DecisionSpec` (model, prompt, tools, infra). Two runs with the
   same fingerprint should produce statistically similar results;
   when they don't, regression detection knows whether to blame the
   agent or the infrastructure.

Everything else — async vs sync, single agent vs HTTP, code grader vs
LLM judge — is a knob you can turn without rewriting your eval set.

That's it. Run `python examples/hello_world.py`, render the JSON report,
then open the sample Markdown report.
