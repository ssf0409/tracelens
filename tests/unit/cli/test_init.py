"""Tests for the `tracelens init` scaffold templates (issue #49)."""

import yaml

from tracelens.cli.init import render_readme, render_workflow, tracelens_requirement


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

    def test_gate_flags_are_present_as_a_commented_block(self):
        text = render_workflow()
        assert "#   --baseline-check" in text
        assert "#   --baselines-file eval/baselines.json" in text
        assert "#   --fail-on-regression moderate" in text

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
        assert "from tracelens import BaselineManager, TaskBaseline" in text
        assert "--baseline-check --baselines-file eval/baselines.json --fail-on-regression moderate" in text
        assert "Prove that it blocks" in text
        assert "0 = gate passed, 1 = blocked, 2 = misconfigured or unevaluable" in text
