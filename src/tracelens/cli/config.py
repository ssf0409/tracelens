"""Project-owned run configuration: ``tracelens run --config tracelens.yaml``.

The file holds exactly what the ``run`` flags hold, so a project can commit
its evaluation command instead of repeating a dozen flags in every CI step
and README. Precedence is fixed:

1. built-in defaults (:data:`RUN_DEFAULTS`),
2. values from the config file,
3. flags given explicitly on the command line.

An omitted flag never replaces a config value with its default, and boolean
flags work in both directions (``--progress`` / ``--no-progress``). Paths in
the file resolve relative to the file; paths on the command line keep the
current-directory semantics they always had. Adapters and graders are
imported from ``run.import_root`` (default: the config file's directory), so
the command behaves the same from any working directory; the process
directory is never changed.

Parsing is strict on purpose: YAML is loaded with the safe loader, duplicate
keys, unknown keys, wrong types, and unsafe constructs are errors before any
agent call, and a config is *only* run configuration -- no profiles,
includes, matrices, or environment interpolation.

Why YAML (and PyYAML): the maintainers chose ``tracelens.yaml`` because it is
what CI systems, downstream projects, and the generated GitHub workflow
already speak. PyYAML is the reference implementation, pure-Python capable,
and used only through ``safe_load`` semantics here.
"""

from __future__ import annotations

import argparse
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# argparse dest -> built-in default. The CLI defaults live here, once.
RUN_DEFAULTS: dict[str, Any] = {
    "eval_set": None,
    "eval_set_format": None,
    "input_field": "input",
    "metadata_fields": None,
    "adapter": None,
    "graders": None,
    "task_ids": None,
    "num_runs": 1,
    "max_concurrency": 5,
    "timeout": 300.0,
    "baseline_check": False,
    "baselines_file": None,
    "require_baselines": False,
    "fail_on_regression": "moderate",
    "output": None,
    "report": None,
    "html_report": None,
    "save_trials": None,
    "progress": False,
    "checkpoint": None,
    "max_infra_retries": 0,
    "infra_exceptions": None,
    "decision_spec": None,
    "noise_band": None,
}

_EVAL_SET_FORMATS = ("json", "jsonl", "csv")
_SEVERITIES = ("minor", "moderate", "severe")


@dataclass(frozen=True)
class _Field:
    key: tuple[str, ...]  # path inside the document, e.g. ("run", "outputs", "results")
    dest: str | None  # argparse dest; None for config-only keys (import_root)
    kind: str  # "str" | "int" | "number" | "bool" | "str_list"
    is_path: bool = False
    choices: tuple[str, ...] | None = None


_FIELDS: tuple[_Field, ...] = (
    _Field(("run", "eval_set"), "eval_set", "str", is_path=True),
    _Field(("run", "eval_set_format"), "eval_set_format", "str", choices=_EVAL_SET_FORMATS),
    _Field(("run", "input_field"), "input_field", "str"),
    _Field(("run", "metadata_fields"), "metadata_fields", "str_list"),
    _Field(("run", "adapter"), "adapter", "str"),
    _Field(("run", "graders"), "graders", "str_list"),
    _Field(("run", "task_ids"), "task_ids", "str_list"),
    _Field(("run", "import_root"), None, "str", is_path=True),
    _Field(("run", "num_runs"), "num_runs", "int"),
    _Field(("run", "max_concurrency"), "max_concurrency", "int"),
    _Field(("run", "timeout"), "timeout", "number"),
    _Field(("run", "progress"), "progress", "bool"),
    _Field(("run", "checkpoint"), "checkpoint", "str", is_path=True),
    _Field(("run", "max_infra_retries"), "max_infra_retries", "int"),
    _Field(("run", "infra_exceptions"), "infra_exceptions", "str_list"),
    _Field(("run", "decision_spec"), "decision_spec", "str", is_path=True),
    _Field(("run", "outputs", "results"), "output", "str", is_path=True),
    _Field(("run", "outputs", "report"), "report", "str", is_path=True),
    _Field(("run", "outputs", "html_report"), "html_report", "str", is_path=True),
    _Field(("run", "outputs", "trials"), "save_trials", "str", is_path=True),
    _Field(("run", "baseline", "enabled"), "baseline_check", "bool"),
    _Field(("run", "baseline", "file"), "baselines_file", "str", is_path=True),
    _Field(("run", "baseline", "fail_on_regression"), "fail_on_regression", "str", choices=_SEVERITIES),
    _Field(("run", "baseline", "require_baselines"), "require_baselines", "bool"),
    _Field(("run", "baseline", "noise_band"), "noise_band", "number"),
)

# Sections (mapping-valued keys) and the leaf keys allowed under each.
_SECTIONS: dict[tuple[str, ...], set[str]] = {}
for _field in _FIELDS:
    _SECTIONS.setdefault(_field.key[:-1], set()).add(_field.key[-1])
for _section in list(_SECTIONS):
    for _depth in range(1, len(_section)):
        _SECTIONS.setdefault(_section[:_depth], set()).add(_section[_depth])
_SECTIONS[()] = {"run"}


class ConfigError(ValueError):
    """A config file could not be used. The message is user-facing."""


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader that refuses duplicate mapping keys instead of keeping the last."""

    def construct_mapping(
        self, node: yaml.MappingNode, deep: bool = False
    ) -> dict[Hashable, Any]:
        seen: set[Hashable] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise ConfigError(
                    f"duplicate key {key!r} at line {key_node.start_mark.line + 1}"
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


@dataclass
class RunConfig:
    """A validated config file: argparse values plus the import root."""

    path: Path
    values: dict[str, Any]
    import_root: Path


def _dotted(key: tuple[str, ...]) -> str:
    return ".".join(key)


def _check_kind(field: _Field, value: Any) -> Any:
    name = _dotted(field.key)
    if field.kind == "bool":
        if not isinstance(value, bool):
            raise ConfigError(f"{name} must be true or false, got {value!r}")
        return value
    if field.kind == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{name} must be an integer, got {value!r}")
        return value
    if field.kind == "number":
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ConfigError(f"{name} must be a number, got {value!r}")
        return float(value)
    if field.kind == "str":
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"{name} must be a non-empty string, got {value!r}")
        if field.choices and value not in field.choices:
            raise ConfigError(
                f"{name} must be one of {', '.join(field.choices)}, got {value!r}"
            )
        return value
    if field.kind == "str_list":
        if not isinstance(value, list) or not value or not all(
            isinstance(item, str) and item.strip() for item in value
        ):
            raise ConfigError(f"{name} must be a non-empty list of strings, got {value!r}")
        return list(value)
    raise ConfigError(f"{name}: unsupported field kind {field.kind!r}")


def _walk(document: Mapping[str, Any], section: tuple[str, ...]) -> None:
    """Reject unknown keys and non-mapping sections, recursively."""
    allowed = _SECTIONS.get(section, set())
    unknown = sorted(str(k) for k in document if k not in allowed)
    if unknown:
        where = _dotted(section) or "top level"
        raise ConfigError(
            f"unknown key(s) under {where}: {', '.join(unknown)}; "
            f"allowed: {', '.join(sorted(allowed))}"
        )
    for key, value in document.items():
        child = (*section, str(key))
        if child in _SECTIONS:
            if not isinstance(value, Mapping):
                raise ConfigError(f"{_dotted(child)} must be a mapping")
            _walk(value, child)


def _lookup(document: Mapping[str, Any], key: tuple[str, ...]) -> tuple[bool, Any]:
    node: Any = document
    for part in key:
        if not isinstance(node, Mapping) or part not in node:
            return False, None
        node = node[part]
    return True, node


def load_run_config(path: str | Path) -> RunConfig:
    """Parse and validate a ``tracelens.yaml`` file.

    Raises:
        ConfigError: Missing or unreadable file, invalid or unsafe YAML,
            duplicate keys, unknown keys, wrong types, or bad enumerations.
            Messages name the file and the dotted key.
    """
    config_path = Path(path)
    try:
        text = config_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {config_path}") from exc
    except OSError as exc:
        raise ConfigError(f"could not read config file {config_path}: {exc}") from exc
    try:
        document = yaml.load(text, Loader=_StrictLoader)  # noqa: S506 - SafeLoader subclass
    except ConfigError as exc:
        raise ConfigError(f"{config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"{config_path}: invalid YAML: {exc}") from exc
    if document is None:
        raise ConfigError(f"{config_path}: the file is empty; expected a 'run:' section")
    if not isinstance(document, Mapping):
        raise ConfigError(f"{config_path}: expected a mapping with a 'run:' section")
    try:
        _walk(document, ())
        if "run" not in document:
            raise ConfigError("missing the 'run:' section")
        values: dict[str, Any] = {}
        import_root = config_path.resolve().parent
        base = config_path.resolve().parent
        for field in _FIELDS:
            present, raw = _lookup(document, field.key)
            if not present:
                continue
            value = _check_kind(field, raw)
            if field.is_path:
                value = str(base / value) if not Path(value).is_absolute() else value
            if field.dest is None:
                import_root = Path(value)
            else:
                values[field.dest] = value
    except ConfigError as exc:
        raise ConfigError(f"{config_path}: {exc}") from exc
    return RunConfig(path=config_path, values=values, import_root=import_root)


def explicit_run_options(args: argparse.Namespace) -> set[str]:
    """The ``run`` settings the user gave explicitly.

    The CLI parser records them in ``args.explicit_run_options``. A namespace
    built by hand (tests, programmatic callers) has no such record, so every
    run setting it carries counts as explicit.
    """
    recorded = getattr(args, "explicit_run_options", None)
    if recorded is None:
        return {dest for dest in RUN_DEFAULTS if hasattr(args, dest)}
    return {dest for dest in recorded if dest in RUN_DEFAULTS}


def resolve_run_settings(
    args: argparse.Namespace,
) -> tuple[argparse.Namespace, Path | None]:
    """Merge defaults, the config file, and explicit flags into one namespace.

    ``args`` is the parsed ``run`` namespace. Flags the user typed are listed
    in ``args.explicit_run_options`` (set by the parser); only those override
    config values. Returns the merged namespace and the import root from the
    config file (``None`` without a config: the current directory is used).

    Raises:
        ConfigError: From :func:`load_run_config`, or when a required setting
            (eval set, adapter, graders) is missing from both sources.
    """
    config = load_run_config(args.config) if getattr(args, "config", None) else None
    merged: dict[str, Any] = dict(RUN_DEFAULTS)
    if config is not None:
        merged.update(config.values)
    explicit = explicit_run_options(args)
    for dest in explicit:
        merged[dest] = getattr(args, dest)
    missing = [
        label
        for label, dest in (
            ("--eval-set (run.eval_set)", "eval_set"),
            ("--adapter (run.adapter)", "adapter"),
            ("--graders (run.graders)", "graders"),
        )
        if not merged.get(dest)
    ]
    if missing:
        source = f" or in {config.path}" if config is not None else ""
        raise ConfigError(
            "missing required setting(s): " + ", ".join(missing)
            + f"; pass them on the command line{source}"
        )
    resolved = argparse.Namespace(**merged)
    resolved.command = "run"
    resolved.config = getattr(args, "config", None)
    resolved.debug = getattr(args, "debug", False)
    resolved.explicit_run_options = set(explicit)
    return resolved, (config.import_root if config is not None else None)
