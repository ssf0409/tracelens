# Installation & Usage Guide

This guide explains how to install and use the `eval-kit` framework in your projects.

## Installation Options

### Option 1: Install from Private GitHub (Recommended)

Since this is a private repository, you can install directly from GitHub:

```bash
# Using uv (recommended)
uv pip install git+https://github.com/ssf0409/eval-kit.git

# Using pip
pip install git+https://github.com/ssf0409/eval-kit.git

# With LLM support (for LLM-based graders)
uv pip install "eval-kit[llm] @ git+https://github.com/ssf0409/eval-kit.git"
```

### Option 2: Add as Dependency in pyproject.toml

Add to your project's `pyproject.toml`:

```toml
[project]
dependencies = [
    "eval-kit @ git+https://github.com/ssf0409/eval-kit.git",
]

# Or with a specific version/tag
dependencies = [
    "eval-kit @ git+https://github.com/ssf0409/eval-kit.git@v0.1.0",
]

# With LLM extras
dependencies = [
    "eval-kit[llm] @ git+https://github.com/ssf0409/eval-kit.git",
]
```

Then install:
```bash
uv sync  # or: pip install -e .
```

### Option 3: Git Submodule (For Development)

If you want to develop both projects together:

```bash
# Add as submodule
git submodule add https://github.com/ssf0409/eval-kit.git libs/eval-kit

# Install in development mode
uv pip install -e libs/eval-kit
```

## Private Repository Authentication

Since `eval-kit` is a private repository, you need to configure authentication for both local development and CI/CD.

### Local Development (SSH)

**Recommended**: Use SSH URLs for local development. This works automatically if you have SSH keys configured with GitHub.

```toml
# In your project's pyproject.toml
[project]
dependencies = [
    "eval-kit @ git+ssh://git@github.com/ssf0409/eval-kit.git",
]
```

Verify SSH is working:
```bash
ssh -T git@github.com
# Should see: "Hi ssf0409! You've successfully authenticated..."
```

### CI/CD Authentication

For GitHub Actions, you have three options:

#### Option A: GITHUB_TOKEN (Same Owner - Easiest)

If all repos are under the same GitHub account (`ssf0409`), use the built-in token:

```yaml
# .github/workflows/eval.yml
jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Configure git for private repos
        run: |
          git config --global url."https://${{ secrets.GITHUB_TOKEN }}@github.com/".insteadOf "git+ssh://git@github.com/"
          git config --global url."https://${{ secrets.GITHUB_TOKEN }}@github.com/".insteadOf "https://github.com/"

      - name: Install uv
        uses: astral-sh/setup-uv@v4

      - name: Install dependencies
        run: uv sync
```

#### Option B: Personal Access Token (Cross-Organization)

For repos in different organizations or accounts:

1. Create a PAT at https://github.com/settings/tokens with `repo` scope
2. Add it as a repository secret: `Settings > Secrets > Actions > PRIVATE_REPO_TOKEN`

```yaml
- name: Configure git for private repos
  run: |
    git config --global url."https://${{ secrets.PRIVATE_REPO_TOKEN }}@github.com/".insteadOf "git+ssh://git@github.com/"
    git config --global url."https://${{ secrets.PRIVATE_REPO_TOKEN }}@github.com/".insteadOf "https://github.com/"
```

#### Option C: Deploy Key (Most Secure)

For production environments, deploy keys provide read-only access scoped to a single repo:

1. Generate a key pair:
   ```bash
   ssh-keygen -t ed25519 -C "eval-kit-deploy" -f eval-kit-deploy -N ""
   ```

2. Add the **public key** (`eval-kit-deploy.pub`) to eval-kit:
   `Settings > Deploy keys > Add deploy key` (enable "Allow read access")

3. Add the **private key** (`eval-kit-deploy`) as a secret in your project:
   `Settings > Secrets > Actions > AGENT_EVAL_DEPLOY_KEY`

4. Use in workflow:
   ```yaml
   - name: Setup SSH for private deps
     uses: webfactory/ssh-agent@v0.9.0
     with:
       ssh-private-key: ${{ secrets.AGENT_EVAL_DEPLOY_KEY }}

   - name: Install dependencies
     run: uv sync
   ```

### Authentication Summary

| Environment | Method | pyproject.toml URL |
|-------------|--------|-------------------|
| Local (macOS/Linux) | SSH keys | `git+ssh://git@github.com/...` |
| GitHub Actions (same owner) | GITHUB_TOKEN | Either SSH or HTTPS (converted) |
| GitHub Actions (different org) | PAT | Either SSH or HTTPS (converted) |
| Production CI | Deploy Key | `git+ssh://git@github.com/...` |

### Recommended Setup

For projects under the same GitHub account:

**pyproject.toml**:
```toml
[project]
dependencies = [
    "eval-kit @ git+ssh://git@github.com/ssf0409/eval-kit.git",
]
```

**GitHub Actions**:
```yaml
- name: Configure git for private repos
  run: |
    git config --global url."https://${{ secrets.GITHUB_TOKEN }}@github.com/".insteadOf "git+ssh://git@github.com/"
```

This gives you:
- Local development uses SSH (your existing git setup)
- CI uses GITHUB_TOKEN (no extra secrets needed)

## Quick Start

### 1. Define a Task

```python
from eval_kit.core.task import Task, TaskExpectation

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
from eval_kit.core.grader import CodeGrader
from eval_kit.core.transcript import Transcript
from eval_kit.core.task import Task

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
from eval_kit.statistics.pass_at_k import PassAtKAnalyzer
from eval_kit.statistics.consistency import ConsistencyAnalyzer

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
from eval_kit.baselines.manager import BaselineManager
from eval_kit.baselines.comparison import RegressionDetector

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
uv run pytest tests/ -v --cov=eval_kit --cov-report=html

# Lint
uv run ruff check src/ tests/

# Type check
uv run mypy src/eval_kit/
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

When integrating `eval-kit` into your project, we recommend this structure:

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
└── pyproject.toml                # Include eval-kit dependency
```

## Next Steps

- [Getting Started](./getting-started.md) — Run your first eval in five minutes
- [Quickstart](./quickstart.md) — Build a custom grader and CLI workflow
- [User Guide](./user-guide.md) — Deep dive into the framework
- [CI/CD Integration Guide](./ci-cd-integration.md) — Automated regression testing
