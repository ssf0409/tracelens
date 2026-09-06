"""Tests for ``tracelens run --config`` (issue #35): loading, validation, precedence."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pytest

from tracelens.cli.config import (
    RUN_DEFAULTS,
    ConfigError,
    explicit_run_options,
    load_run_config,
    resolve_run_settings,
)
from tracelens.cli.main import build_parser

FULL_CONFIG = """\
run:
  eval_set: eval/tasks.json
  eval_set_format: json
  input_field: prompt
  metadata_fields: [difficulty, area]
  adapter: eval.adapter.StarterAdapter
  graders: [eval.grader.StarterGrader, eval.grader.OtherGrader]
  task_ids: [t-1, t-2]
  import_root: src
  num_runs: 3
  max_concurrency: 2
  timeout: 45
  progress: true
  checkpoint: eval/results/checkpoint.json
  max_infra_retries: 2
  infra_exceptions: [builtins.OSError]
  decision_spec: eval/spec.json
  outputs:
    results: eval/results/results.json
    report: eval/results/report.md
    html_report: eval/results/report.html
    trials: eval/results/trials.json
  baseline:
    enabled: true
    file: eval/baselines.json
    fail_on_regression: severe
    require_baselines: true
    noise_band: 0.05
"""

MINIMAL = "run:\n  eval_set: t.json\n  adapter: a.A\n  graders: [g.G]\n"


def _write(tmp_path: Path, text: str, name: str = "tracelens.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _parse(*argv: str) -> argparse.Namespace:
    return build_parser().parse_args(["run", *argv])


class TestLoadRunConfig:
    def test_every_key_maps_to_its_flag_and_paths_resolve_against_the_file(
        self, tmp_path: Path
    ) -> None:
        nested = tmp_path / "project" / "conf"
        nested.mkdir(parents=True)
        path = _write(nested, FULL_CONFIG)
        config = load_run_config(path)
        base = nested.resolve()
        assert config.path == path
        assert config.import_root == base / "src"
        assert config.values == {
            "eval_set": str(base / "eval/tasks.json"),
            "eval_set_format": "json",
            "input_field": "prompt",
            "metadata_fields": ["difficulty", "area"],
            "adapter": "eval.adapter.StarterAdapter",
            "graders": ["eval.grader.StarterGrader", "eval.grader.OtherGrader"],
            "task_ids": ["t-1", "t-2"],
            "num_runs": 3,
            "max_concurrency": 2,
            "timeout": 45.0,
            "progress": True,
            "checkpoint": str(base / "eval/results/checkpoint.json"),
            "max_infra_retries": 2,
            "infra_exceptions": ["builtins.OSError"],
            "decision_spec": str(base / "eval/spec.json"),
            "output": str(base / "eval/results/results.json"),
            "report": str(base / "eval/results/report.md"),
            "html_report": str(base / "eval/results/report.html"),
            "save_trials": str(base / "eval/results/trials.json"),
            "baseline_check": True,
            "baselines_file": str(base / "eval/baselines.json"),
            "fail_on_regression": "severe",
            "require_baselines": True,
            "noise_band": 0.05,
        }
        # Every config value lands on a real run setting, and the file covers
        # every setting the CLI has (import_root is config-only).
        assert set(config.values) == set(RUN_DEFAULTS)

    def test_absolute_paths_are_kept(self, tmp_path: Path) -> None:
        target = tmp_path / "elsewhere" / "abs.json"
        path = _write(tmp_path, f"run:\n  eval_set: {target}\n")
        assert load_run_config(path).values["eval_set"] == str(target)

    def test_import_root_defaults_to_the_config_directory(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "run:\n  eval_set: tasks.json\n")
        assert load_run_config(path).import_root == tmp_path.resolve()

    def test_minimal_file_yields_only_the_keys_present(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "run:\n  num_runs: 2\n")
        assert load_run_config(path).values == {"num_runs": 2}

    @pytest.mark.parametrize(
        ("text", "fragment"),
        [
            ("run:\n  adapters: x\n", "unknown key(s) under run: adapters"),
            ("run: {}\nprofiles: {}\n", "unknown key(s) under top level: profiles"),
            ("run:\n  outputs:\n    json: a\n", "unknown key(s) under run.outputs: json"),
            (
                "run:\n  baseline:\n    threshold: 1\n",
                "unknown key(s) under run.baseline: threshold",
            ),
            ("run:\n  outputs: results.json\n", "run.outputs must be a mapping"),
            ("run:\n  num_runs: two\n", "run.num_runs must be an integer"),
            ("run:\n  num_runs: true\n", "run.num_runs must be an integer"),
            ("run:\n  num_runs: 2.5\n", "run.num_runs must be an integer"),
            ("run:\n  timeout: fast\n", "run.timeout must be a number"),
            ("run:\n  progress: yes please\n", "run.progress must be true or false"),
            ("run:\n  eval_set: ''\n", "run.eval_set must be a non-empty string"),
            ("run:\n  eval_set: [a]\n", "run.eval_set must be a non-empty string"),
            (
                "run:\n  graders: eval.grader.G\n",
                "run.graders must be a non-empty list of strings",
            ),
            ("run:\n  graders: []\n", "run.graders must be a non-empty list of strings"),
            ("run:\n  graders: [1]\n", "run.graders must be a non-empty list of strings"),
            (
                "run:\n  eval_set_format: xml\n",
                "run.eval_set_format must be one of json, jsonl, csv",
            ),
            (
                "run:\n  baseline:\n    fail_on_regression: high\n",
                "run.baseline.fail_on_regression must be one of minor, moderate, severe",
            ),
            ("", "the file is empty"),
            ("- a\n- b\n", "expected a mapping with a 'run:' section"),
            ("other: 1\n", "unknown key(s) under top level: other"),
            ("{}\n", "missing the 'run:' section"),
            ("run: ~\n", "run must be a mapping"),
            ("run:\n  num_runs: 1\n  num_runs: 2\n", "duplicate key 'num_runs' at line 3"),
            (
                "run:\n  eval_set: !!python/object/apply:os.system [echo]\n",
                "invalid YAML",
            ),
            ("run:\n  eval_set: [unclosed\n", "invalid YAML"),
        ],
    )
    def test_invalid_files_are_rejected_naming_the_file_and_key(
        self, tmp_path: Path, text: str, fragment: str
    ) -> None:
        path = _write(tmp_path, text)
        with pytest.raises(ConfigError) as exc_info:
            load_run_config(path)
        message = str(exc_info.value)
        assert message.startswith(str(path)), message
        assert fragment in message, message

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="config file not found"):
            load_run_config(tmp_path / "nope.yaml")

    def test_config_error_is_a_value_error(self) -> None:
        assert issubclass(ConfigError, ValueError)


class TestResolveRunSettings:
    def test_file_values_fill_in_and_the_rest_are_the_built_in_defaults(
        self, tmp_path: Path
    ) -> None:
        config = _write(tmp_path, MINIMAL)
        resolved, import_root = resolve_run_settings(_parse("--config", str(config)))
        assert import_root == tmp_path.resolve()
        assert resolved.eval_set == str(tmp_path.resolve() / "t.json")
        assert resolved.adapter == "a.A"
        assert resolved.graders == ["g.G"]
        for dest, default in RUN_DEFAULTS.items():
            if dest not in ("eval_set", "adapter", "graders"):
                assert getattr(resolved, dest) == default, dest
        assert resolved.config == str(config)
        assert resolved.command == "run"
        assert resolved.explicit_run_options == set()

    def test_explicit_flags_override_the_file_and_omitted_flags_do_not(
        self, tmp_path: Path
    ) -> None:
        config = _write(tmp_path, FULL_CONFIG)
        resolved, _ = resolve_run_settings(
            _parse(
                "--config", str(config),
                "--num-runs", "1",  # equal to the built-in default, still explicit
                "--timeout", "0.5",
                "--graders", "cli.G",
                "--output", "cli-results.json",  # a CLI path keeps cwd semantics
                "--fail-on-regression", "minor",
                "--no-progress",  # explicit false over a true in the file
                "--no-require-baselines",
                "--max-infra-retries", "0",  # explicit zero over a 2 in the file
            )
        )
        assert resolved.num_runs == 1
        assert resolved.timeout == 0.5
        assert resolved.graders == ["cli.G"]
        assert resolved.output == "cli-results.json"
        assert resolved.fail_on_regression == "minor"
        assert resolved.progress is False
        assert resolved.require_baselines is False
        assert resolved.max_infra_retries == 0
        # Omitted flags keep the file's values, not argparse defaults.
        assert resolved.max_concurrency == 2
        assert resolved.baseline_check is True
        assert resolved.adapter == "eval.adapter.StarterAdapter"
        assert resolved.report == str(tmp_path.resolve() / "eval/results/report.md")
        assert resolved.noise_band == 0.05
        assert resolved.explicit_run_options == {
            "num_runs", "timeout", "graders", "output", "fail_on_regression",
            "progress", "require_baselines", "max_infra_retries",
        }

    @pytest.mark.parametrize(
        ("in_file", "flag", "expected"),
        [
            ("true", "--no-progress", False),
            ("false", "--progress", True),
            ("true", "--progress", True),
            ("false", "--no-progress", False),
        ],
    )
    def test_boolean_flags_override_in_both_directions(
        self, tmp_path: Path, in_file: str, flag: str, expected: bool
    ) -> None:
        config = _write(tmp_path, MINIMAL + f"  progress: {in_file}\n")
        resolved, _ = resolve_run_settings(_parse("--config", str(config), flag))
        assert resolved.progress is expected

    def test_no_baseline_check_switches_a_configured_gate_off(self, tmp_path: Path) -> None:
        config = _write(
            tmp_path, MINIMAL + "  baseline:\n    enabled: true\n    file: b.json\n"
        )
        on, _ = resolve_run_settings(_parse("--config", str(config)))
        off, _ = resolve_run_settings(_parse("--config", str(config), "--no-baseline-check"))
        assert on.baseline_check is True
        assert off.baseline_check is False
        assert off.baselines_file == str(tmp_path.resolve() / "b.json")

    def test_flags_alone_behave_as_before(self) -> None:
        resolved, import_root = resolve_run_settings(
            _parse("--eval-set", "t.json", "--adapter", "a.A", "--graders", "g.G", "h.H",
                   "--progress")
        )
        assert import_root is None
        assert resolved.eval_set == "t.json"  # cwd-relative, untouched
        assert resolved.graders == ["g.G", "h.H"]
        assert resolved.progress is True
        assert resolved.num_runs == 1
        assert resolved.config is None
        assert resolved.explicit_run_options == {"eval_set", "adapter", "graders", "progress"}

    @pytest.mark.parametrize(
        ("argv", "missing"),
        [
            ([], "--eval-set (run.eval_set), --adapter (run.adapter), --graders (run.graders)"),
            (["--eval-set", "t.json", "--adapter", "a.A"], "--graders (run.graders)"),
        ],
    )
    def test_missing_required_settings_are_named(self, argv: list[str], missing: str) -> None:
        with pytest.raises(ConfigError) as exc_info:
            resolve_run_settings(_parse(*argv))
        assert str(exc_info.value) == (
            f"missing required setting(s): {missing}; pass them on the command line"
        )

    def test_missing_settings_name_the_config_file_too(self, tmp_path: Path) -> None:
        config = _write(tmp_path, "run:\n  eval_set: t.json\n")
        expected = (
            "missing required setting(s): --adapter (run.adapter), --graders (run.graders); "
            f"pass them on the command line or in {config}"
        )
        with pytest.raises(ConfigError, match=re.escape(expected)):
            resolve_run_settings(_parse("--config", str(config)))

    def test_config_errors_propagate(self, tmp_path: Path) -> None:
        config = _write(tmp_path, "run:\n  bogus: 1\n")
        with pytest.raises(ConfigError, match="unknown key"):
            resolve_run_settings(_parse("--config", str(config)))

    def test_hand_built_namespace_counts_every_setting_as_explicit(self) -> None:
        args = argparse.Namespace(eval_set="t.json", adapter="a.A", graders=["g.G"], num_runs=4)
        assert explicit_run_options(args) == {"eval_set", "adapter", "graders", "num_runs"}
        resolved, import_root = resolve_run_settings(args)
        assert resolved.num_runs == 4
        assert resolved.max_concurrency == RUN_DEFAULTS["max_concurrency"]
        assert import_root is None


class TestRunParser:
    def test_omitted_flags_stay_out_of_the_namespace(self) -> None:
        args = _parse("--num-runs", "2")
        assert vars(args) == {"command": "run", "debug": False, "config": None, "num_runs": 2}

    def test_help_spells_out_defaults_without_leaking_suppress(self) -> None:
        subparsers = build_parser()._subparsers
        assert subparsers is not None
        run_parser = next(
            action.choices["run"]
            for action in subparsers._group_actions
            if isinstance(action, argparse._SubParsersAction)
        )
        text = run_parser.format_help()
        assert "SUPPRESS" not in text
        assert "--config FILE" in text
        assert "--no-progress" in text
        assert "--no-baseline-check" in text
        assert "--no-require-baselines" in text
        # Defaults are spelled out in the help strings (argparse cannot
        # substitute %(default)s when the default is SUPPRESS).
        actions = {action.dest: action for action in run_parser._actions}
        assert "(default: 5)" in (actions["max_concurrency"].help or "")
        assert "(default: moderate)" in (actions["fail_on_regression"].help or "")
        assert "(default: 300)" in (actions["timeout"].help or "")
        for action in run_parser._actions:
            assert "%(default)" not in (action.help or ""), action.dest
