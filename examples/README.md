# TraceLens Examples

Run these examples from the repository root after installing TraceLens:

```bash
uv pip install -e ".[dev,http,llm]"
```

## Example Ladder

| Step | File | What It Shows |
|------|------|---------------|
| 1 | `hello_world.py` | The smallest possible local eval: task, adapter, grader, runner. |
| 2 | `contract_eval.py` | Generate graders from a behavior contract. |
| 3 | `http_agent_eval.py` | Evaluate an agent exposed as an HTTP JSON endpoint. |
| 4 | `noise_aware_regression.py` | Compare runs with different infrastructure fingerprints. |

## Hello World

```bash
python examples/hello_world.py
```

Expected output:

```text
tracelens hello-world
--------------------
trials run : 3
pass rate  : 100%
```

Use this file as the template when you want to evaluate a normal Python
function or local agent loop.

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

## Coverage Notes

These four examples are intentionally small and dependency-light. They are
enough to teach the core framework and support the first public release.

Future examples should focus on scenarios that are documented but not yet
represented as runnable scripts:

- LLM-as-judge using a fake or recorded provider.
- Multi-step tool-use transcript review.
- Human calibration against grader output.
- Downstream project CI that installs TraceLens from PyPI.
