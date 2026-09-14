"""Scaffold a starter TraceLens eval directory."""

import argparse
import json
import re
import sys
from pathlib import Path

from tracelens import _version

STARTER_TASKS: dict[str, object] = {
    "tasks": [
        {
            "task_id": "starter-capital",
            "name": "Answer a simple geography question",
            "input_data": {
                "question": "What is the capital of France?",
            },
            "expectation": {
                "expected_output": {
                    "answer": "Paris",
                },
            },
            "tags": ["starter"],
        },
        {
            "task_id": "starter-math",
            "name": "Answer a simple arithmetic question",
            "input_data": {
                "question": "What is 2 + 2?",
            },
            "expectation": {
                "expected_output": {
                    "answer": "4",
                },
            },
            "tags": ["starter"],
        },
    ]
}


ADAPTER_TEMPLATE = '''"""Starter adapter for a TraceLens eval suite.

Replace ``starter_agent`` with a call into your real agent.
"""

from typing import Any

from tracelens import SimpleAdapter

# Canned responses for the starter eval set. A real agent receives input_data
# and generates answers dynamically; this lookup keeps the scaffold runnable
# offline before you wire in your real model or code.
CANNED_ANSWERS: dict[str, str] = {
    "What is the capital of France?": "Paris",
    "What is 2 + 2?": "4",
}


async def starter_agent(input_data: dict[str, Any]) -> dict[str, Any]:
    """Return a deterministic answer for the starter tasks.

    This keeps the generated eval runnable before you wire in a real agent.
    """
    question = input_data.get("question", "")
    return {"answer": CANNED_ANSWERS.get(question, "unknown")}


class StarterAdapter(SimpleAdapter):
    """No-argument adapter loadable by ``tracelens run``."""

    # Declared identity recorded in every run's provenance. Uncomment and bump
    # it when the agent under test changes (attribution evidence, not proof
    # that the code is identical).
    # provenance_version = "starter-1"

    def __init__(self) -> None:
        super().__init__(starter_agent)
'''


GRADER_TEMPLATE = '''"""Starter grader for a TraceLens eval suite."""

from typing import Any

from tracelens import CodeGrader, Outcome, Task, Transcript


def _get_expected(task: Task) -> str | None:
    if task.expectation is None or task.expectation.expected_output is None:
        return None
    expected_output = task.expectation.expected_output
    if isinstance(expected_output, dict):
        val = expected_output.get("answer")
        return str(val) if val is not None else None
    return str(expected_output)


class StarterGrader(CodeGrader):
    """Passes when ``final_output["answer"]`` matches task expectation."""

    # Declared identity recorded in every run's provenance. Uncomment and bump
    # it when the rubric changes: a changed grader is a different measurement,
    # so `tracelens compare` refuses to compare runs across it.
    # provenance_version = "starter-1"

    def __init__(self) -> None:
        super().__init__("starter")

    def compute_metrics(self, transcript: Transcript, task: Task) -> dict[str, float]:
        expected = _get_expected(task)
        if expected is None:
            return {"exact_match": 0.0}
        final_output = transcript.final_output
        if not isinstance(final_output, dict) or "answer" not in final_output:
            return {"exact_match": 0.0}
        actual = str(final_output["answer"]).strip().lower()
        return {"exact_match": float(actual == expected.strip().lower())}

    def determine_pass(
        self,
        metrics: dict[str, float],
        task: Task,
    ) -> tuple[bool, float]:
        score = metrics.get("exact_match", 0.0)
        return score == 1.0, score

    async def grade(self, transcript: Transcript, task: Task) -> Outcome:
        metrics = self.compute_metrics(transcript, task)
        passed, score = self.determine_pass(metrics, task)

        expected = _get_expected(task)
        final_output = transcript.final_output

        if expected is None:
            feedback = "task declares no expected answer"
        elif final_output is None:
            feedback = f"expected {expected!r}, got null output"
        elif not isinstance(final_output, dict):
            feedback = f"expected dict output, got {type(final_output).__name__}"
        elif "answer" not in final_output:
            keys = sorted(final_output.keys())
            feedback = f"expected key 'answer' in output, got keys {keys!r}"
        elif not passed:
            actual = final_output["answer"]
            feedback = f"expected {expected!r}, got {actual!r}"
        else:
            feedback = None

        return self.create_outcome(
            trial_id=transcript.task_id,
            passed=passed,
            score=score,
            metrics=metrics,
            feedback=feedback,
        )
'''


CONFIG_TEMPLATE = """# Generated by `tracelens init`. Run the suite from any directory with:
#
#     tracelens run --config tracelens.yaml
#
# Every key is a `tracelens run` flag (`tracelens run --help` lists them); a
# flag given on the command line overrides the value here. Paths resolve
# relative to this file, and `eval.adapter` / `eval.grader` are imported from
# this file's directory. Keep secrets in environment variables, not here.
run:
  eval_set: eval/tasks.json
  adapter: eval.adapter.StarterAdapter
  graders:
    - eval.grader.StarterGrader
  num_runs: 1
  outputs:
    results: eval/results/results.json
    report: eval/results/report.md
    html_report: eval/results/report.html
    trials: eval/results/trials.json
  # Step 4 of eval/README.md: once eval/baselines.json exists, uncomment this
  # block to block pull requests on regressions. CI runs the same file.
  # baseline:
  #   enabled: true
  #   file: eval/baselines.json
  #   fail_on_regression: moderate
"""


README_TEMPLATE = """# TraceLens Starter Eval

Generated by `tracelens init`. Everything under `eval/` is yours to edit;
`tracelens init . --force` rewrites untouched starter files while preserving
files you have edited (pass `--overwrite-edited` to replace edited files too).

## 1. Run the starter suite

```bash
tracelens run --config tracelens.yaml
```

`tracelens.yaml` (next to `eval/`) holds the run settings: the eval set, the
adapter and grader import paths, and where the results go
(`eval/results/`). Every key is a `tracelens run` flag, and a flag on the
command line overrides the file, so
`tracelens run --config tracelens.yaml --num-runs 3` repeats each task three
times without editing anything. The command works from any directory when
given the path to the file.

The starter agent returns canned answers for the starter questions, so this
passes by construction: it proves the wiring, not your agent.

## 2. What the CI workflow does

`.github/workflows/eval.yml` runs the same `tracelens run --config
tracelens.yaml` on every pull request to `main` (and on manual dispatch),
posts the Markdown report to the job summary, and uploads the results as
artifacts. Until you enable the gate in step 4 it is an integration smoke
test: it fails only if the run itself errors.

Installation in CI:

- an existing uv project is installed from `uv.lock` (`uv sync --frozen`),
  so CI matches your local environment;
- a bare repository gets a fresh environment;
- TraceLens is installed only if the project does not already provide it,
  pinned to `__REQUIREMENT__`. Bump the pin on purpose.

Every pull request is evaluated by default so agent code changes cannot skip
the eval. To evaluate only when specific paths change, uncomment the
`paths:` block in the workflow and list your agent's source directories.

## 3. Make it yours

1. Replace `starter_agent` in `eval/adapter.py` with a call into your agent.
2. Replace `StarterGrader` in `eval/grader.py` with your acceptance criteria.
3. Replace the tasks in `eval/tasks.json` with real cases, ideally drawn from
   past failures.
4. Keep `tracelens.yaml` pointing at them. It is the one place your shell and
   CI read the run settings from, so a change there applies to both.

## 4. Enable the regression gate

The gate compares each task with a stored baseline and blocks the pull
request on a regression.

1. Run the suite on a version you trust (step 1), then store baselines from
   that run and commit `eval/baselines.json`:

   ```bash
   python - <<'EOF'
   import json
   from pathlib import Path

   from tracelens import BaselineManager, TaskBaseline

   results = json.loads(Path("eval/results/results.json").read_text())
   manager = BaselineManager("eval/baselines.json")
   for task in results["task_summaries"]:
       # task_hash lets the gate refuse to compare a task whose content changed
       baseline = TaskBaseline(task_id=task["task_id"], task_hash=task.get("task_hash"))
       baseline.add_metric(
           "pass_rate", task["pass_rate"], std=0.05, sample_size=task["num_trials"]
       )
       manager.set_baseline(baseline)
   manager.save()
   EOF
   ```

2. In `tracelens.yaml`, uncomment the `baseline:` block:

   ```yaml
   baseline:
     enabled: true
     file: eval/baselines.json
     fail_on_regression: moderate
   ```

   CI picks this up on the next push because the workflow runs the same
   file. For a one-off check without editing it, the equivalent flags are
   `--baseline-check --baselines-file eval/baselines.json --fail-on-regression moderate`.

3. Prove that it blocks: make `starter_agent` return a wrong answer, run
   `tracelens run --config tracelens.yaml`, and confirm it exits 1 with
   `BLOCKED` in `eval/results/report.md`. To see the failure itself, run
   `tracelens inspect eval/results/trials.json --failures --eval-set eval/tasks.json`:
   it prints the wrong answer next to the task and the grader's verdict.
   Revert the change; the same command exits 0 again.

Exit codes: 0 = gate passed, 1 = blocked, 2 = misconfigured or unevaluable
(for example a task with no gradable trials). The decision is recorded in
`eval/results/results.json` under `gate`; see
https://ssf0409.github.io/tracelens/ci-cd-integration/#reading-the-gate.

Docs: https://ssf0409.github.io/tracelens/getting-started/
"""


WORKFLOW_TEMPLATE = """name: TraceLens Evaluation

# Generated by `tracelens init`. Until the regression gate is enabled (see
# eval/README.md, step 4) this is an integration smoke test: it proves the
# eval suite runs on every pull request and fails only if the run errors.

on:
  pull_request:
    branches: [main]
    # Every pull request is evaluated so agent code changes never skip the
    # eval. Narrow this only once you know which paths matter, for example:
    #   paths:
    #     - "src/**"
    #     - "eval/**"
    #     - "pyproject.toml"
    #     - "uv.lock"
    #     - "tracelens.yaml"
  workflow_dispatch:

permissions:
  contents: read

jobs:
  eval:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v6

      - uses: astral-sh/setup-uv@11f9893b081a58869d3b5fccaea48c9e9e46f990 # v8.3.2

      - name: Set up Python
        run: uv python install 3.12

      # Existing uv project: install from the lockfile so CI matches your
      # local environment. Bare repository: create a fresh environment.
      # TraceLens is installed only if the project does not already provide
      # it, pinned to the release that generated this file.
      - name: Install dependencies
        run: |
          if [ -f uv.lock ]; then
            uv sync --frozen
          elif [ -f pyproject.toml ]; then
            uv sync
          else
            uv venv --python 3.12
          fi
          if ! .venv/bin/python -c "import tracelens" 2>/dev/null; then
            uv pip install "__REQUIREMENT__"
          fi

      # The same command you run locally. Outputs land in eval/results/ and
      # the regression gate is switched on in tracelens.yaml (eval/README.md,
      # step 4), so this step never needs editing.
      - name: Run TraceLens starter eval
        run: .venv/bin/tracelens run --config tracelens.yaml

      # The report exists only after a successful preflight; a missing file
      # must not turn a clear TraceLens error into a `cat` failure.
      - name: Add report to job summary
        if: always()
        run: |
          if [ -f eval/results/report.md ]; then
            cat eval/results/report.md >> "$GITHUB_STEP_SUMMARY"
          else
            echo "No TraceLens report was written; see the run step for the error." \\
              >> "$GITHUB_STEP_SUMMARY"
          fi

      - name: Upload evaluation artifacts
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: tracelens-results
          if-no-files-found: ignore
          path: |
            eval/results/results.json
            eval/results/report.md
            eval/results/report.html
            eval/results/trials.json
"""


def tracelens_requirement(version: str | None = None) -> str:
    """The TraceLens requirement pinned into generated files.

    A released version (``X.Y.Z``) is pinned exactly so the workflow installs
    the same TraceLens that generated it. A development or unknown version
    cannot be installed from PyPI, so the requirement falls back to
    ``tracelens`` unpinned; the generated README says to pin it.
    """
    current = _version.__version__ if version is None else version
    if re.fullmatch(r"\d+\.\d+\.\d+", current):
        return f"tracelens=={current}"
    return "tracelens"


def render_readme(requirement: str | None = None) -> str:
    """The generated ``eval/README.md`` for the given TraceLens requirement."""
    return README_TEMPLATE.replace("__REQUIREMENT__", requirement or tracelens_requirement())


def render_workflow(requirement: str | None = None) -> str:
    """The generated ``.github/workflows/eval.yml`` for the given requirement."""
    return WORKFLOW_TEMPLATE.replace(
        "__REQUIREMENT__", requirement or tracelens_requirement()
    )


def add_init_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Add the 'init' subcommand to the CLI."""
    parser = subparsers.add_parser(
        "init",
        help="Scaffold a runnable eval/ directory, tracelens.yaml, and CI workflow",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Project directory to initialize (default: current directory)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite untouched generated files; skip edited ones",
    )
    parser.add_argument(
        "--overwrite-edited",
        action="store_true",
        help="Overwrite edited files too, saving a .bak copy first",
    )


def _starter_files() -> dict[Path, str]:
    return {
        Path("eval/__init__.py"): '"""Starter TraceLens eval package."""\n',
        Path("eval/tasks.json"): json.dumps(STARTER_TASKS, indent=2) + "\n",
        Path("eval/adapter.py"): ADAPTER_TEMPLATE,
        Path("eval/grader.py"): GRADER_TEMPLATE,
        Path("eval/README.md"): render_readme(),
        Path("tracelens.yaml"): CONFIG_TEMPLATE,
        Path(".github/workflows/eval.yml"): render_workflow(),
    }


def cmd_init(args: argparse.Namespace) -> int:
    """Execute the 'init' subcommand."""
    root = Path(args.path)
    files = {root / relative: content for relative, content in _starter_files().items()}

    conflicts = [path for path in files if path.exists()]
    if conflicts and not (args.force or args.overwrite_edited):
        print("Error: refusing to overwrite existing files:", file=sys.stderr)
        for path in conflicts:
            print(f"  {path}", file=sys.stderr)
        print("Re-run with --force to overwrite generated files.", file=sys.stderr)
        return 2

    for path, content in files.items():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            continue

        try:
            existing = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            existing = None

        if existing == content:
            path.write_text(content, encoding="utf-8")
        elif args.overwrite_edited:
            backup = path.parent / f"{path.name}.bak"
            if backup.exists():
                counter = 1
                while (path.parent / f"{path.name}.bak.{counter}").exists():
                    counter += 1
                backup = path.parent / f"{path.name}.bak.{counter}"
            if existing is not None:
                backup.write_text(existing, encoding="utf-8")
            else:
                backup.write_bytes(path.read_bytes())
            path.write_text(content, encoding="utf-8")
            print(f"overwrote {path} (backed up to {backup})")
        else:
            print(f"kept {path} (edited); pass --overwrite-edited to replace it")

    print(f"Initialized TraceLens eval scaffold in {root}")
    print(f"Next: tracelens run --config {root / 'tracelens.yaml'}")
    return 0
