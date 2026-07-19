# Installation

How to install `tracelens` and add it to your own project. Once it's installed,
head to [Getting Started](./getting-started.md) to run your first eval.

## Installation Options

### Option 1: Install from PyPI

```bash
# Using uv (recommended)
uv pip install tracelens

# Using pip
pip install tracelens

# With LLM support (for LLM-based graders)
uv pip install "tracelens[llm]"
```

For release mechanics, see [Releasing TraceLens](releasing.md).

### Option 2: Add as a Dependency in pyproject.toml

Add to your project's `pyproject.toml`:

```toml
[project]
dependencies = [
    "tracelens>=0.3.0",
]

# With LLM extras
dependencies = [
    "tracelens[llm]>=0.3.0",
]
```

Then install:
```bash
uv sync  # or: pip install -e .
```

### Option 3: Local Development Checkout

If you want to contribute to TraceLens itself:

```bash
git clone https://github.com/ssf0409/tracelens.git
cd tracelens

# Install with development tools
uv venv
uv pip install -e ".[dev,http,llm]"
```

See [Contributor Testing](contributor-testing.md) for the local verification
gate (`make verify`) and the full testing tiers.

## Optional Extras

- `tracelens[http]` installs `httpx` for `HTTPAPIAdapter`.
- `tracelens[llm]` installs the OpenAI and Anthropic SDKs for custom
  `LLMProvider` subclasses.
- `tracelens[datasets]` installs Hugging Face `datasets` for the optional
  `HFDatasetLoader`.
- `tracelens[dev]` installs pytest, ruff, mypy, and type stubs for
  contributors.

Extras compose normally:

```bash
uv pip install "tracelens[http,llm,datasets]"
```

## CI Installation

For GitHub Actions, install your project dependencies normally. If your
project depends on `tracelens>=0.3.0`, `uv sync` or `pip install -e .`
is enough; no extra repository checkout or authentication is required.

```yaml
jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - name: Set up Python
        run: uv python install 3.12
      - name: Install dependencies
        run: uv sync
      - name: Run evaluation
        run: |
          uv run tracelens run \
            --eval-set eval/tasks.json \
            --adapter myproject.eval.adapters.CIAgentAdapter \
            --graders myproject.eval.graders.CIQualityGrader \
            --num-runs 3 \
            --output eval/results/results.json \
            --report eval/results/report.md
```

The full PR-gating workflow (baselines, regression blocking, report artifacts)
lives in [CI/CD Integration](ci-cd-integration.md).

## Verify the Install

```bash
python -c "import tracelens; print(tracelens.__version__)"
tracelens --help
```

From a repository checkout, verify the example/report path:

```bash
python examples/hello_world.py
tracelens report --results examples/reports/hello_world_report.json --format markdown
```

## Project Structure for Integration

When integrating `tracelens` into your project, we recommend this structure:

```
your-project/
├── eval/
│   ├── __init__.py
│   ├── tasks/                    # Task definitions
│   │   ├── __init__.py
│   │   ├── task_schema.py        # Your Task subclass
│   │   └── scenarios/            # JSON task files
│   │       ├── scenario_001.json
│   │       └── scenario_002.json
│   ├── graders/                  # Grader implementations
│   │   ├── __init__.py
│   │   ├── quality_grader.py
│   │   └── domain_grader.py
│   ├── baselines/
│   │   └── baselines.json        # Stored baselines
│   ├── harness.py                # Evaluation orchestrator
│   └── conftest.py               # Test fixtures
├── .github/
│   └── workflows/
│       └── eval.yml              # CI evaluation workflow
└── pyproject.toml                # Include tracelens dependency
```

## Next Steps

- [Getting Started (5 min)](./getting-started.md) — run your first eval and meet
  the four-piece skeleton.
- [Build Your First Eval](./quickstart.md) — write your own Task, grader, and
  CLI workflow.
- [Core Concepts & Glossary](./concepts.md) — the pipeline and every object in
  one page.
- [CI/CD Integration](./ci-cd-integration.md) — automated regression testing.
