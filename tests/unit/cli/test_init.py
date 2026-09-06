"""Tests for the `tracelens init` scaffold templates (issues #49, #35)."""

from pathlib import Path

import yaml

from tracelens.cli.config import load_run_config
from tracelens.cli.init import (
    ADAPTER_TEMPLATE,
    CONFIG_TEMPLATE,
    GRADER_TEMPLATE,
    render_readme,
    render_workflow,
    tracelens_requirement,
)


def enable_gate_block(config_text: str) -> str:
    """Do what eval/README.md step 4.2 says: uncomment the ``baseline:`` block."""
    lines = config_text.splitlines(keepends=True)
    start = lines.index("  # baseline:\n")
    lines[start] = "  baseline:\n"
    for i in range(start + 1, len(lines)):
        if not lines[i].startswith("  #   "):
            break
        lines[i] = "    " + lines[i][len("  #   "):]
    return "".join(lines)


def _triggers(workflow: dict) -> dict:
    # PyYAML reads the bare key ``on`` as boolean True (YAML 1.1).
    return workflow.get("on", workflow.get(True))


class TestWorkflowTemplate:
    def test_is_valid_yaml_with_expected_shape(self):
        data = yaml.safe_load(render_workflow("tracelens==1.2.3"))
        triggers = _triggers(data)
        assert triggers["pull_request"] == {"branches": ["main"]}
        assert "paths" not in triggers["pull_request"]
        assert "workflow_dispatch" in triggers
        assert data["permissions"] == {"contents": "read"}
        steps = data["jobs"]["eval"]["steps"]
        assert steps[0]["uses"] == "actions/checkout@v6"
        assert steps[1]["uses"].startswith("astral-sh/setup-uv@11f9893b")

    def test_install_step_is_reproducible_and_pins_tracelens(self):
        text = render_workflow("tracelens==1.2.3")
        assert "uv sync --frozen" in text
        assert "uv venv --python 3.12" in text
        assert 'uv pip install "tracelens==1.2.3"' in text
        assert 'python -c "import tracelens"' in text  # only when missing
        assert ".venv/bin/tracelens run" in text

    def test_run_step_is_the_documented_config_command(self):
        data = yaml.safe_load(render_workflow())
        steps = {s.get("name"): s for s in data["jobs"]["eval"]["steps"]}
        run = steps["Run TraceLens starter eval"]["run"].strip()
        assert run == ".venv/bin/tracelens run --config tracelens.yaml"

    def test_gate_lives_in_the_config_not_the_workflow(self):
        assert "--baseline-check" not in render_workflow()
        assert "  # baseline:\n  #   enabled: true\n" in CONFIG_TEMPLATE

    def test_summary_and_artifact_steps_tolerate_missing_files(self):
        data = yaml.safe_load(render_workflow())
        steps = {s.get("name"): s for s in data["jobs"]["eval"]["steps"]}
        summary = steps["Add report to job summary"]
        assert summary["if"] == "always()"
        assert "if [ -f eval/results/report.md ]" in summary["run"]
        assert "No TraceLens report was written" in summary["run"]
        upload = steps["Upload evaluation artifacts"]
        assert upload["uses"] == "actions/upload-artifact@v4"
        assert upload["with"]["if-no-files-found"] == "ignore"

    def test_paths_filter_is_offered_only_as_a_comment(self):
        text = render_workflow()
        assert "#   paths:" in text
        assert "\n    paths:" not in text


class TestRequirementPin:
    def test_release_versions_are_pinned_exactly(self):
        assert tracelens_requirement("0.4.0") == "tracelens==0.4.0"

    def test_dev_and_unknown_versions_fall_back_to_unpinned(self):
        assert tracelens_requirement("0.1.1.dev129") == "tracelens"
        assert tracelens_requirement("0.0.0+unknown") == "tracelens"

    def test_readme_and_workflow_share_the_requirement(self):
        assert "pinned to `tracelens==9.9.9`" in render_readme("tracelens==9.9.9")
        assert 'uv pip install "tracelens==9.9.9"' in render_workflow("tracelens==9.9.9")


class TestConfigTemplate:
    def test_parses_into_the_run_the_readme_describes(self, tmp_path: Path):
        path = tmp_path / "tracelens.yaml"
        path.write_text(CONFIG_TEMPLATE)
        config = load_run_config(path)
        base = tmp_path.resolve()
        assert config.import_root == base
        assert config.values == {
            "eval_set": str(base / "eval/tasks.json"),
            "adapter": "eval.adapter.StarterAdapter",
            "graders": ["eval.grader.StarterGrader"],
            "num_runs": 1,
            "output": str(base / "eval/results/results.json"),
            "report": str(base / "eval/results/report.md"),
            "html_report": str(base / "eval/results/report.html"),
            "save_trials": str(base / "eval/results/trials.json"),
        }

    def test_uncommenting_the_gate_block_enables_the_documented_gate(self, tmp_path: Path):
        path = tmp_path / "tracelens.yaml"
        path.write_text(enable_gate_block(CONFIG_TEMPLATE))
        values = load_run_config(path).values
        assert values["baseline_check"] is True
        assert values["baselines_file"] == str(tmp_path.resolve() / "eval/baselines.json")
        assert values["fail_on_regression"] == "moderate"

    def test_documents_the_command_and_keeps_secrets_out(self):
        assert "tracelens run --config tracelens.yaml" in CONFIG_TEMPLATE
        assert "environment variables" in CONFIG_TEMPLATE
        assert "key" not in CONFIG_TEMPLATE.lower().replace("every key", "").replace("api key", "x")


class TestReadmeTemplate:
    def test_walkthrough_sections_and_gate_recipe(self):
        text = render_readme()
        for heading in (
            "## 1. Run the starter suite",
            "## 2. What the CI workflow does",
            "## 3. Make it yours",
            "## 4. Enable the regression gate",
        ):
            assert heading in text
        assert "tracelens run --config tracelens.yaml" in text
        assert "--adapter eval.adapter.StarterAdapter" not in text  # one documented run command
        assert "tracelens inspect eval/results/trials.json --failures" in text
        assert "from tracelens import BaselineManager, TaskBaseline" in text
        assert "uncomment the `baseline:` block" in text
        assert "--baseline-check --baselines-file eval/baselines.json --fail-on-regression moderate" in text
        assert "Prove that it blocks" in text
        assert "0 = gate passed, 1 = blocked, 2 = misconfigured or unevaluable" in text


class TestProvenanceVersionHint:
    """Issue #76: the scaffold shows where a declared identity goes."""

    def test_templates_carry_the_commented_declaration(self):
        for template in (ADAPTER_TEMPLATE, GRADER_TEMPLATE):
            assert '    # provenance_version = "starter-1"\n' in template
            # Commented out on purpose: the scaffold declares no version until the user does.
            assert "\n    provenance_version" not in template
