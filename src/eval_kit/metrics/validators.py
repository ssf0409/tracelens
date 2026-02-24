"""Validator metric graders for checking output structure and content.

Provides deterministic CodeGraders that validate agent outputs against
schemas, required content, regex patterns, and arbitrary constraints.
"""

from __future__ import annotations

import re
from typing import Any

from eval_kit.core.grader import CodeGrader, EvalPolicy, GraderConfig
from eval_kit.core.task import Task
from eval_kit.core.transcript import Transcript
from eval_kit.execution.registry import load_class

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_JSON_TYPE_MAP: dict[str, type | tuple[type, ...]] = {
    "object": dict,
    "array": list,
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "null": type(None),
}


def _basic_type_check(data: Any, schema: dict[str, Any]) -> list[str]:
    """Minimal JSON Schema type checking when ``jsonschema`` is unavailable.

    Only validates the top-level ``type`` and ``required`` keywords.
    Returns a list of error messages (empty means valid).
    """
    errors: list[str] = []

    expected_type = schema.get("type")
    if expected_type and expected_type in _JSON_TYPE_MAP:
        allowed = _JSON_TYPE_MAP[expected_type]
        if not isinstance(data, allowed):
            errors.append(
                f"Expected top-level type '{expected_type}', "
                f"got {type(data).__name__}"
            )
            return errors

    if expected_type == "object" and isinstance(data, dict):
        for key in schema.get("required", []):
            if key not in data:
                errors.append(f"Missing required field: '{key}'")

    return errors


# ===========================================================================
# JsonSchemaGrader
# ===========================================================================


class JsonSchemaGrader(CodeGrader):
    """Validate ``transcript.final_output`` against a JSON Schema.

    Uses the ``jsonschema`` library when available; otherwise falls back to
    basic top-level type and required-field checking.

    Default policy: GATE (schema violations block CI).
    """

    def __init__(
        self,
        grader_id: str,
        *,
        schema: dict[str, Any],
        config: GraderConfig | None = None,
    ) -> None:
        if config is None:
            config = GraderConfig(policy=EvalPolicy.GATE)
        super().__init__(grader_id, config=config)
        self.schema = schema

    def compute_metrics(
        self,
        transcript: Transcript,
        task: Task,
    ) -> dict[str, float]:
        data = transcript.final_output
        errors: list[str] = []

        try:
            import jsonschema as _js  # noqa: N813

            if _js is None:
                raise ImportError
            _js.validate(instance=data, schema=self.schema)
        except ImportError:
            errors = _basic_type_check(data, self.schema)
        except Exception as exc:  # noqa: BLE001
            errors = [str(exc)]

        return {
            "schema_valid": 1.0 if not errors else 0.0,
            "error_count": float(len(errors)),
        }

    def determine_pass(
        self,
        metrics: dict[str, float],
        task: Task,
    ) -> tuple[bool, float]:
        passed = metrics["schema_valid"] == 1.0
        score = metrics["schema_valid"]
        return passed, score


# ===========================================================================
# StructuredOutputGrader
# ===========================================================================


class StructuredOutputGrader(CodeGrader):
    """Validate ``transcript.final_output`` by parsing it with a Pydantic model.

    The model is loaded at grading time via ``eval_kit.execution.registry.load_class``
    from a dotted path such as ``"myproject.models.ResponseSchema"``.

    Default policy: GATE.
    """

    def __init__(
        self,
        grader_id: str,
        *,
        model_path: str,
        config: GraderConfig | None = None,
    ) -> None:
        if config is None:
            config = GraderConfig(policy=EvalPolicy.GATE)
        super().__init__(grader_id, config=config)
        self.model_path = model_path

    def compute_metrics(
        self,
        transcript: Transcript,
        task: Task,
    ) -> dict[str, float]:
        data = transcript.final_output

        try:
            model_cls = load_class(self.model_path)
        except (ImportError, AttributeError):
            return {"parse_valid": 0.0, "validation_errors": 1.0}

        if not isinstance(data, dict):
            return {"parse_valid": 0.0, "validation_errors": 1.0}

        try:
            model_cls.model_validate(data)
            return {"parse_valid": 1.0, "validation_errors": 0.0}
        except Exception as exc:  # noqa: BLE001
            error_count = 1.0
            if hasattr(exc, "error_count"):
                error_count = float(exc.error_count())
            elif hasattr(exc, "errors"):
                err_list = exc.errors()
                if isinstance(err_list, list):
                    error_count = float(len(err_list))
            return {"parse_valid": 0.0, "validation_errors": error_count}

    def determine_pass(
        self,
        metrics: dict[str, float],
        task: Task,
    ) -> tuple[bool, float]:
        passed = metrics["parse_valid"] == 1.0
        score = metrics["parse_valid"]
        return passed, score


# ===========================================================================
# ContainsGrader
# ===========================================================================


class ContainsGrader(CodeGrader):
    """Check whether ``str(transcript.final_output)`` contains required strings
    and does not contain forbidden strings.

    Default policy: TRACK.
    """

    def __init__(
        self,
        grader_id: str,
        *,
        required: list[str],
        forbidden: list[str] | None = None,
        config: GraderConfig | None = None,
    ) -> None:
        if config is None:
            config = GraderConfig(policy=EvalPolicy.TRACK)
        super().__init__(grader_id, config=config)
        self.required = required
        self.forbidden = forbidden or []

    def compute_metrics(
        self,
        transcript: Transcript,
        task: Task,
    ) -> dict[str, float]:
        text = str(transcript.final_output)

        required_hits = sum(1 for r in self.required if r in text)
        required_ratio = (
            required_hits / len(self.required) if self.required else 1.0
        )

        forbidden_hits = sum(1 for f in self.forbidden if f in text)

        return {
            "required_found": required_ratio,
            "forbidden_found": float(forbidden_hits),
        }

    def determine_pass(
        self,
        metrics: dict[str, float],
        task: Task,
    ) -> tuple[bool, float]:
        all_required = metrics["required_found"] == 1.0
        no_forbidden = metrics["forbidden_found"] == 0.0
        passed = all_required and no_forbidden
        score = metrics["required_found"] if no_forbidden else 0.0
        return passed, score


# ===========================================================================
# RegexMatchGrader
# ===========================================================================


class RegexMatchGrader(CodeGrader):
    """Check whether ``str(transcript.final_output)`` matches each regex pattern.

    Default policy: TRACK.
    """

    def __init__(
        self,
        grader_id: str,
        *,
        patterns: list[str],
        config: GraderConfig | None = None,
    ) -> None:
        if config is None:
            config = GraderConfig(policy=EvalPolicy.TRACK)
        super().__init__(grader_id, config=config)
        self.patterns = patterns

    def compute_metrics(
        self,
        transcript: Transcript,
        task: Task,
    ) -> dict[str, float]:
        text = str(transcript.final_output)
        matched = sum(1 for p in self.patterns if re.search(p, text))
        ratio = matched / len(self.patterns) if self.patterns else 1.0
        return {"patterns_matched": ratio}

    def determine_pass(
        self,
        metrics: dict[str, float],
        task: Task,
    ) -> tuple[bool, float]:
        passed = metrics["patterns_matched"] == 1.0
        score = metrics["patterns_matched"]
        return passed, score


# ===========================================================================
# ConstraintGrader
# ===========================================================================


class ConstraintGrader(CodeGrader):
    """Evaluate a list of heterogeneous constraints against the agent output.

    Supported constraint types:
    - ``must_include``: ``str(output)`` must contain the value
    - ``must_not_include``: ``str(output)`` must not contain the value
    - ``numeric_range``: ``output[field]`` must be within [min, max]
    - ``enum``: ``output[field]`` must be one of the allowed values

    Default policy: GATE.
    """

    def __init__(
        self,
        grader_id: str,
        *,
        constraints: list[dict[str, Any]],
        config: GraderConfig | None = None,
    ) -> None:
        if config is None:
            config = GraderConfig(policy=EvalPolicy.GATE)
        super().__init__(grader_id, config=config)
        self.constraints = constraints

    def compute_metrics(
        self,
        transcript: Transcript,
        task: Task,
    ) -> dict[str, float]:
        if not self.constraints:
            return {"constraints_met": 1.0, "violations": 0.0}

        data = transcript.final_output
        text = str(data)
        met = 0
        violations = 0

        for constraint in self.constraints:
            ctype = constraint.get("type")

            if ctype == "must_include":
                if constraint.get("value", "") in text:
                    met += 1
                else:
                    violations += 1

            elif ctype == "must_not_include":
                if constraint.get("value", "") not in text:
                    met += 1
                else:
                    violations += 1

            elif ctype == "numeric_range":
                field = constraint.get("field", "")
                if isinstance(data, dict) and field in data:
                    val = data[field]
                    lo = constraint.get("min", float("-inf"))
                    hi = constraint.get("max", float("inf"))
                    if isinstance(val, (int, float)) and lo <= val <= hi:
                        met += 1
                    else:
                        violations += 1
                else:
                    violations += 1

            elif ctype == "enum":
                field = constraint.get("field", "")
                allowed = constraint.get("values", [])
                if isinstance(data, dict) and field in data:
                    if data[field] in allowed:
                        met += 1
                    else:
                        violations += 1
                else:
                    violations += 1

            else:
                # Unknown constraint type counts as violation
                violations += 1

        total = len(self.constraints)
        return {
            "constraints_met": met / total,
            "violations": float(violations),
        }

    def determine_pass(
        self,
        metrics: dict[str, float],
        task: Task,
    ) -> tuple[bool, float]:
        passed = metrics["constraints_met"] == 1.0 and metrics["violations"] == 0.0
        score = metrics["constraints_met"]
        return passed, score
