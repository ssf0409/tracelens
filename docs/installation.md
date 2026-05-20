# Installation & Usage Guide

This guide explains how to install and use the `tracelens` framework in your projects.

## Installation Options

### Option 1: Install from GitHub

Until the first PyPI release is published, install directly from GitHub:

```bash
# Using uv (recommended)
uv pip install git+https://github.com/ssf0409/tracelens.git

# Using pip
pip install git+https://github.com/ssf0409/tracelens.git

# With LLM support (for LLM-based graders)
uv pip install "tracelens[llm] @ git+https://github.com/ssf0409/tracelens.git"
```

After PyPI publishing, the install command will become:

```bash
uv pip install tracelens
```

For release mechanics, see [Releasing TraceLens](releasing.md).

### Option 2: Add as a Dependency in pyproject.toml

Add to your project's `pyproject.toml`:

```toml
[project]
dependencies = [
    "tracelens @ git+https://github.com/ssf0409/tracelens.git",
]

# Or with a specific version/tag
dependencies = [
    "tracelens @ git+https://github.com/ssf0409/tracelens.git@v0.1.0",
]

# With LLM extras
dependencies = [
    "tracelens[llm] @ git+https://github.com/ssf0409/tracelens.git",
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

## Optional Extras

- `tracelens[http]` installs `httpx` for `HTTPAPIAdapter`.
- `tracelens[llm]` installs the OpenAI and Anthropic SDKs for custom
  `LLMProvider` subclasses.
- `tracelens[dev]` installs pytest, ruff, mypy, and type stubs for
  contributors.

Extras compose normally:

```bash
uv pip install "tracelens[http,llm] @ git+https://github.com/ssf0409/tracelens.git"
```

## CI Installation

For GitHub Actions, install your project dependencies normally. If your
project depends on tracelens from GitHub, `uv sync` or `pip install -e .`
is enough; no extra repository authentication is required for a public
repository.

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
        run: uv run python -m eval.harness --baseline-check
```

## Verify the Install

```bash
python -c "import tracelens; print(tracelens.__version__)"
tracelens --help
python examples/hello_world.py
```

## Quick Start

### 1. Define a Task

```python
from tracelens.core.task import Task, TaskExpectation

task = Task(
    task_id="goal-decomposition-001",
    name="Learn Python Goal",
    description="Decompose a goal to learn Python programming",
    input_data={
        "goal": "Learn Python programming in 3 months",
        "user_context": {
            "experience": "beginner",
            "hours_per_week": 10,
        },
    },
    expectation=TaskExpectation(
        expected_metrics={"quality": 0.8, "specificity": 0.7},
    ),
    category="programming",
    tags=["python", "beginner"],
)
```

### 2. Create a Grader

```python
from tracelens.core.grader import CodeGrader
from tracelens.core.transcript import Transcript
from tracelens.core.task import Task

class QualityGrader(CodeGrader):
    """Grade based on output quality metrics."""

    grader_id = "quality_grader"
    grader_version = "1.0.0"

    def compute_metrics(
        self,
        transcript: Transcript,
        task: Task
    ) -> dict[str, float]:
        output = transcript.final_output

        # Compute your quality metrics
        return {
            "num_phases": len(output.get("phases", [])),
            "has_timeline": 1.0 if "timeline" in output else 0.0,
            "has_resources": 1.0 if "resources" in output else 0.0,
        }

    def determine_pass(
        self,
        metrics: dict[str, float],
        task: Task
    ) -> tuple[bool, float]:
        # Compute overall score and pass/fail
        score = (
            min(metrics["num_phases"] / 3, 1.0) * 0.5 +
            metrics["has_timeline"] * 0.25 +
            metrics["has_resources"] * 0.25
        )
        passed = score >= 0.7
        return passed, score
```

### 3. Run Evaluation and Compute Statistics

```python
from tracelens.statistics.pass_at_k import PassAtKAnalyzer
from tracelens.statistics.consistency import ConsistencyAnalyzer

# Collect results from multiple runs
pass_results = {
    "task1": [True, True, False, True, True],
    "task2": [True, False, True, True, True],
}

# Compute pass@k (capability)
pak_analyzer = PassAtKAnalyzer(k_values=[1, 3, 5])
capability = pak_analyzer.analyze(pass_results)
print(capability)  # {"pass@1": 0.7, "pass@3": 0.95, "pass@5": 0.99}

# Compute pass^k (reliability)
consistency_analyzer = ConsistencyAnalyzer(k_values=[2, 3, 5])
reliability = consistency_analyzer.analyze(pass_results)
print(reliability)  # {"pass^2": 0.6, "pass^3": 0.4, "pass^5": 0.2}
```

### 4. Baseline Comparison

```python
from tracelens.baselines.manager import BaselineManager
from tracelens.baselines.comparison import RegressionDetector

# Load/create baseline manager
manager = BaselineManager("baselines.json")

# Update baseline from current results
manager.update_baseline(
    task_id="goal-decomposition",
    metrics={"quality": 0.85, "specificity": 0.78},
)
manager.save()

# Compare new results to baseline
detector = RegressionDetector(min_delta_percent=5.0)
baseline = manager.get_baseline("goal-decomposition")

current_results = [
    {"quality": 0.82, "specificity": 0.75},
    {"quality": 0.80, "specificity": 0.73},
]

report = detector.compare(baseline, current_results)

if report.should_block_ci():
    print("REGRESSION DETECTED!")
    print(report.to_ci_output())
```

## Development Commands

Using uv (recommended):

```bash
# Install with dev dependencies
uv pip install -e ".[dev]"

# Run tests
uv run pytest tests/ -v

# Run tests with coverage
uv run pytest tests/ -v --cov=tracelens --cov-report=html

# Lint
uv run ruff check src/ tests/

# Type check
uv run mypy src/tracelens/
```

Using Docker:

```bash
# Run tests in container
docker compose run --rm test

# Run with coverage
docker compose run --rm test-coverage

# Interactive shell
docker compose run --rm dev
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

- [Getting Started](./getting-started.md) — Run your first eval in five minutes
- [Quickstart](./quickstart.md) — Build a custom grader and CLI workflow
- [User Guide](./user-guide.md) — Deep dive into the framework
- [CI/CD Integration Guide](./ci-cd-integration.md) — Automated regression testing
