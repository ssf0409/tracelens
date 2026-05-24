# TraceLens Examples

Run these examples from the repository root after installing TraceLens:

```bash
uv pip install -e ".[dev]"
```

## Example Ladder

| Step | File | What It Shows |
|------|------|---------------|
| 1 | `hello_world.py` | The smallest possible local eval: task, adapter, grader, runner. |
| 2 | `contract_eval.py` | Generate graders from a behavior contract. |
| 3 | `http_agent_eval.py` | Evaluate an agent exposed as an HTTP JSON endpoint. |
| 4 | `noise_aware_regression.py` | Compare runs with different infrastructure fingerprints. |
| 5 | `llm_provider_examples.py` | Wire OpenAI and Anthropic SDK clients into `LLMGrader` through `LLMProvider`. |

## Hello World

```bash
python examples/hello_world.py
tracelens report --results examples/reports/hello_world_report.json --format markdown
```

Expected output:

```text
tracelens hello-world
--------------------
trials run : 9
pass rate  : 100%
report json: examples/reports/hello_world_report.json
sample md  : examples/reports/hello_world_report.md
```

Use this file as the template when you want to evaluate a normal Python
function or local agent loop. The generated sample report at
`examples/reports/hello_world_report.md` shows tasks, trials, pass@k,
pass^k, graders, baseline comparison, regression result, and CI summary.

## HTTP Agent

```bash
python examples/http_agent_eval.py
```

This starts a local stdlib HTTP server, evaluates it with `HTTPAPIAdapter`, and
grades the JSON response shape.

## Contract Eval

```bash
python examples/contract_eval.py
```

This is the fastest way to encode strict output rules without writing every
grader by hand.

## Noise-Aware Regression

```bash
python examples/noise_aware_regression.py
```

This demonstrates how TraceLens separates agent regressions from small
infrastructure-driven differences.

## OpenAI and Anthropic Provider Examples

```bash
python examples/llm_provider_examples.py
```

This runs in dry-run mode by default, using recorded judge responses so no API
keys or network calls are required. It shows one `LLMGrader` quality dimension,
`instruction_following`, and the provider subclass pattern TraceLens expects.

For live calls, install the optional LLM dependencies and set one provider:

```bash
uv pip install -e ".[llm]"
OPENAI_API_KEY=... python examples/llm_provider_examples.py --provider openai --live
ANTHROPIC_API_KEY=... python examples/llm_provider_examples.py --provider anthropic --live
```

CLI options:

- `--provider`: `openai`, `anthropic`, or `all`; defaults to all providers.
- `--live`: make live SDK calls; omitted means dry-run mode.

Environment variables can be used as optional fallbacks:

- `TRACELENS_PROVIDER`: used when `--provider` is omitted.
- `TRACELENS_LIVE`: set to `1` to make live SDK calls when `--live` is omitted.
- `OPENAI_API_KEY`: required for `--provider openai --live`.
- `ANTHROPIC_API_KEY`: required for `--provider anthropic --live`.

Live mode fails with a clear error if the requested provider API key or optional
SDK dependency is missing.

## Coverage Notes

These examples are intentionally small and dependency-light. They are
enough to teach the core framework and support the first public release.

Future examples should focus on scenarios that are documented but not yet
represented as runnable scripts:

- Multi-step tool-use transcript review.
- Human calibration against grader output.
- Downstream project CI that installs TraceLens from PyPI.
