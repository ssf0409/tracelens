"""Validator metric graders for checking output structure and content.

Provides deterministic CodeGraders that validate agent outputs against
schemas, required content, regex patterns, and arbitrary constraints.
"""

from __future__ import annotations

import json
import re
from typing import Any

import jsonschema
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from tracelens.core.grader import CodeGrader, EvalPolicy, GraderConfig
from tracelens.core.task import Task
from tracelens.core.transcript import Transcript
from tracelens.execution.registry import load_class

_VALID_CONSTRAINT_TYPES = {"must_include", "must_not_include", "numeric_range", "enum"}


def _serialize_output(val: Any) -> str:
    """Serialize output value to string safely.

    Dicts and lists are serialized as JSON with sorted keys to ensure deterministic
    representation; other non-str primitives are converted using str(val).
    """
    if isinstance(val, (dict, list)):
        return json.dumps(val, default=str, ensure_ascii=False, sort_keys=True)
    return str(val)


# ===========================================================================
# JsonSchemaGrader
# ===========================================================================


class JsonSchemaGrader(CodeGrader):
    """Validate ``transcript.final_output`` against a JSON Schema.

    Uses the ``jsonschema`` library for full schema validation. Selects the
    appropriate validator class based on the schema's ``$schema`` declaration,
    falling back to Draft 7.

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
        validator_cls = jsonschema.validators.validator_for(
            schema, default=jsonschema.Draft7Validator
        )
        try:
            validator_cls.check_schema(schema)
        except jsonschema.SchemaError as exc:
            raise ValueError(
                f"JsonSchemaGrader '{grader_id}': invalid schema: {exc.message}"
            ) from exc
        self.schema = schema
        self._validator = validator_cls(schema)

    def compute_metrics(
        self,
        transcript: Transcript,
        task: Task,
    ) -> dict[str, float]:
        data = transcript.final_output
        errors = [e.message for e in self._validator.iter_errors(data)]

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

    The model is loaded at grading time via ``tracelens.execution.registry.load_class``
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
            loaded = load_class(self.model_path)
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                f"StructuredOutputGrader '{self.grader_id}' cannot load model "
                f"'{self.model_path}': {exc}. Verify the dotted path is correct "
                f"and the module is importable."
            ) from exc

        if not (isinstance(loaded, type) and issubclass(loaded, BaseModel)):
            raise RuntimeError(
                f"StructuredOutputGrader '{self.grader_id}' loaded "
                f"'{self.model_path}' but it is not a pydantic.BaseModel subclass."
            )
        model_cls: type[BaseModel] = loaded

        if not isinstance(data, dict):
            return {"parse_valid": 0.0, "validation_errors": 1.0}

        try:
            model_cls.model_validate(data)
            return {"parse_valid": 1.0, "validation_errors": 0.0}
        except PydanticValidationError as exc:
            return {"parse_valid": 0.0, "validation_errors": float(exc.error_count())}

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
    """Check whether ``transcript.final_output`` contains required strings
    and does not contain forbidden strings.

    If ``field`` is specified and ``final_output`` is a dict, checks the value
    of ``final_output[field]``. If ``final_output`` is None, grading fails with
    ``output_present=0.0``.

    Default policy: TRACK.
    """

    def __init__(
        self,
        grader_id: str,
        *,
        required: list[str],
        forbidden: list[str] | None = None,
        field: str | None = None,
        config: GraderConfig | None = None,
    ) -> None:
        if config is None:
            config = GraderConfig(policy=EvalPolicy.TRACK)
        super().__init__(grader_id, config=config)
        self.required = required
        self.forbidden = forbidden or []
        self.field = field

    def compute_metrics(
        self,
        transcript: Transcript,
        task: Task,
    ) -> dict[str, float]:
        raw_output = transcript.final_output
        if raw_output is None:
            return {
                "required_found": 0.0,
                "forbidden_found": 0.0,
                "output_present": 0.0,
            }

        target = raw_output
        if self.field is not None and isinstance(raw_output, dict):
            target = raw_output.get(self.field)
            if target is None:
                return {
                    "required_found": 0.0,
                    "forbidden_found": 0.0,
                    "output_present": 0.0,
                }

        text = _serialize_output(target)

        required_hits = sum(1 for r in self.required if r in text)
        required_ratio = (
            required_hits / len(self.required) if self.required else 1.0
        )

        forbidden_hits = sum(1 for f in self.forbidden if f in text)

        return {
            "required_found": required_ratio,
            "forbidden_found": float(forbidden_hits),
            "output_present": 1.0,
        }

    def determine_pass(
        self,
        metrics: dict[str, float],
        task: Task,
    ) -> tuple[bool, float]:
        output_ok = metrics.get("output_present", 1.0) == 1.0
        all_required = metrics["required_found"] == 1.0
        no_forbidden = metrics["forbidden_found"] == 0.0
        passed = output_ok and all_required and no_forbidden
        score = metrics["required_found"] if (output_ok and no_forbidden) else 0.0
        return passed, score


# ===========================================================================
# RegexMatchGrader
# ===========================================================================


class RegexMatchGrader(CodeGrader):
    """Check whether ``transcript.final_output`` matches each regex pattern.

    If ``field`` is specified and ``final_output`` is a dict, checks the value
    of ``final_output[field]``. If ``final_output`` is None, grading fails with
    ``output_present=0.0``.

    Default policy: TRACK.
    """

    def __init__(
        self,
        grader_id: str,
        *,
        patterns: list[str],
        field: str | None = None,
        config: GraderConfig | None = None,
    ) -> None:
        if config is None:
            config = GraderConfig(policy=EvalPolicy.TRACK)
        super().__init__(grader_id, config=config)
        for i, p in enumerate(patterns):
            try:
                re.compile(p)
            except re.error as exc:
                raise ValueError(
                    f"RegexMatchGrader '{grader_id}': pattern[{i}] is invalid: "
                    f"'{p}' -- {exc}"
                ) from exc
        self.patterns = patterns
        self.field = field

    def compute_metrics(
        self,
        transcript: Transcript,
        task: Task,
    ) -> dict[str, float]:
        raw_output = transcript.final_output
        if raw_output is None:
            return {"patterns_matched": 0.0, "output_present": 0.0}

        target = raw_output
        if self.field is not None and isinstance(raw_output, dict):
            target = raw_output.get(self.field)
            if target is None:
                return {"patterns_matched": 0.0, "output_present": 0.0}

        text = _serialize_output(target)
        matched = sum(1 for p in self.patterns if re.search(p, text))
        ratio = matched / len(self.patterns) if self.patterns else 1.0
        return {"patterns_matched": ratio, "output_present": 1.0}

    def determine_pass(
        self,
        metrics: dict[str, float],
        task: Task,
    ) -> tuple[bool, float]:
        output_ok = metrics.get("output_present", 1.0) == 1.0
        passed = output_ok and metrics["patterns_matched"] == 1.0
        score = metrics["patterns_matched"] if output_ok else 0.0
        return passed, score


# ===========================================================================
# ConstraintGrader
# ===========================================================================


class ConstraintGrader(CodeGrader):
    """Evaluate a list of heterogeneous constraints against the agent output.

    Supported constraint types:
    - ``must_include``: stringified output must contain ``value`` (str)
    - ``must_not_include``: stringified output must not contain ``value`` (str)
    - ``numeric_range``: ``output[field]`` must be within [min, max]
    - ``enum``: ``output[field]`` must be one of the allowed ``values`` (list)

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
        for i, c in enumerate(constraints):
            ctype = c.get("type")
            if ctype not in _VALID_CONSTRAINT_TYPES:
                raise ValueError(
                    f"ConstraintGrader '{grader_id}': constraint[{i}] has "
                    f"unknown type '{ctype}'. "
                    f"Valid types: {sorted(_VALID_CONSTRAINT_TYPES)}"
                )
            if ctype in ("must_include", "must_not_include"):
                if "value" not in c or not isinstance(c["value"], str):
                    raise ValueError(
                        f"ConstraintGrader '{grader_id}': constraint[{i}] of type '{ctype}' "
                        f"requires 'value' of type str, got {type(c.get('value')).__name__}"
                    )
            elif ctype == "numeric_range":
                if "field" not in c or not isinstance(c["field"], str):
                    raise ValueError(
                        f"ConstraintGrader '{grader_id}': constraint[{i}] of type 'numeric_range' "
                        f"requires 'field' of type str"
                    )
                lo = c.get("min", float("-inf"))
                hi = c.get("max", float("inf"))
                if not isinstance(lo, (int, float)) or isinstance(lo, bool):
                    raise ValueError(
                        f"ConstraintGrader '{grader_id}': constraint[{i}] 'min' must be numeric"
                    )
                if not isinstance(hi, (int, float)) or isinstance(hi, bool):
                    raise ValueError(
                        f"ConstraintGrader '{grader_id}': constraint[{i}] 'max' must be numeric"
                    )
                if lo > hi:
                    raise ValueError(
                        f"ConstraintGrader '{grader_id}': constraint[{i}] 'min' ({lo}) > 'max' ({hi})"
                    )
            elif ctype == "enum":
                if "field" not in c or not isinstance(c["field"], str):
                    raise ValueError(
                        f"ConstraintGrader '{grader_id}': constraint[{i}] of type 'enum' "
                        f"requires 'field' of type str"
                    )
                if "values" not in c or not isinstance(c["values"], list):
                    raise ValueError(
                        f"ConstraintGrader '{grader_id}': constraint[{i}] of type 'enum' "
                        f"requires 'values' of type list"
                    )
        self.constraints = constraints

    def compute_metrics(
        self,
        transcript: Transcript,
        task: Task,
    ) -> dict[str, float]:
        if not self.constraints:
            return {"constraints_met": 1.0, "violations": 0.0}

        data = transcript.final_output
        if data is None:
            total = len(self.constraints)
            return {
                "constraints_met": 0.0,
                "violations": float(total),
            }

        text = _serialize_output(data)
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
                    if isinstance(val, (int, float)) and not isinstance(val, bool) and lo <= val <= hi:
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
