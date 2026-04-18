"""Tests for BehaviorContract schema and to_graders() generation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from eval_kit.contracts.contract import BehaviorContract
from eval_kit.core.grader import EvalPolicy, Grader


class TestBehaviorContractSchema:
    """Tests for BehaviorContract Pydantic model."""

    def test_minimal_contract(self) -> None:
        contract = BehaviorContract(contract_id="test", version="1.0")
        assert contract.contract_id == "test"
        assert contract.version == "1.0"
        assert contract.tools_allowed == []
        assert contract.tools_required == []
        assert contract.must_include == []

    def test_full_contract(self) -> None:
        contract = BehaviorContract(
            contract_id="planner-v1",
            version="1.0.0",
            output_schema={"type": "object", "properties": {"result": {"type": "string"}}},
            tools_allowed=["search", "calculator"],
            tools_required=["search"],
            max_tokens=5000,
            max_latency_ms=3000.0,
            must_include=["disclaimer"],
            must_not_include=["internal_key"],
            custom_constraints=[
                {"type": "numeric_range", "field": "confidence", "min": 0.0, "max": 1.0},
            ],
        )
        assert contract.contract_id == "planner-v1"
        assert len(contract.tools_allowed) == 2
        assert contract.max_tokens == 5000

    def test_contract_serialization_roundtrip(self) -> None:
        contract = BehaviorContract(
            contract_id="test",
            version="1.0",
            output_schema={"type": "object"},
            tools_required=["search"],
            max_tokens=1000,
            must_include=["hello"],
        )
        data = contract.model_dump(mode="json")
        restored = BehaviorContract.model_validate(data)
        assert restored.contract_id == contract.contract_id
        assert restored.output_schema == contract.output_schema
        assert restored.tools_required == contract.tools_required


class TestBehaviorContractToGraders:
    """Tests for to_graders() auto-generation."""

    def test_empty_contract_produces_no_graders(self) -> None:
        contract = BehaviorContract(contract_id="empty", version="1.0")
        graders = contract.to_graders()
        assert graders == []

    def test_output_schema_produces_json_schema_grader(self) -> None:
        contract = BehaviorContract(
            contract_id="test",
            version="1.0",
            output_schema={"type": "object", "required": ["result"]},
        )
        graders = contract.to_graders()

        schema_graders = [g for g, p in graders if "json_schema" in g.grader_id]
        assert len(schema_graders) == 1
        # Should default to GATE policy
        policies = [p for g, p in graders if "json_schema" in g.grader_id]
        assert policies[0] == EvalPolicy.GATE

    def test_tool_constraints_produce_tool_call_grader(self) -> None:
        contract = BehaviorContract(
            contract_id="test",
            version="1.0",
            tools_allowed=["search", "calc"],
            tools_required=["search"],
        )
        graders = contract.to_graders()

        tool_graders = [g for g, p in graders if "tool_call" in g.grader_id]
        assert len(tool_graders) == 1
        policies = [p for g, p in graders if "tool_call" in g.grader_id]
        assert policies[0] == EvalPolicy.GATE

    def test_budget_constraints_produce_budget_graders(self) -> None:
        contract = BehaviorContract(
            contract_id="test",
            version="1.0",
            max_tokens=5000,
            max_latency_ms=3000.0,
        )
        graders = contract.to_graders()

        grader_ids = [g.grader_id for g, _ in graders]
        assert any("latency" in gid for gid in grader_ids)
        assert any("token" in gid for gid in grader_ids)
        # Budget graders default to WARN
        for g, p in graders:
            assert p == EvalPolicy.WARN

    def test_content_constraints_produce_contains_grader(self) -> None:
        contract = BehaviorContract(
            contract_id="test",
            version="1.0",
            must_include=["disclaimer", "terms"],
            must_not_include=["secret_key"],
        )
        graders = contract.to_graders()

        contains_graders = [g for g, p in graders if "contains" in g.grader_id]
        assert len(contains_graders) == 1

    def test_custom_constraints_produce_constraint_grader(self) -> None:
        contract = BehaviorContract(
            contract_id="test",
            version="1.0",
            custom_constraints=[
                {"type": "numeric_range", "field": "score", "min": 0.0, "max": 1.0},
                {"type": "enum", "field": "status", "values": ["ok", "error"]},
            ],
        )
        graders = contract.to_graders()

        constraint_graders = [g for g, p in graders if "constraint" in g.grader_id]
        assert len(constraint_graders) == 1
        policies = [p for g, p in graders if "constraint" in g.grader_id]
        assert policies[0] == EvalPolicy.GATE

    def test_full_contract_produces_all_grader_types(self) -> None:
        contract = BehaviorContract(
            contract_id="full",
            version="1.0",
            output_schema={"type": "object"},
            tools_required=["search"],
            tools_allowed=["search", "calc"],
            max_tokens=5000,
            max_latency_ms=3000.0,
            must_include=["disclaimer"],
            custom_constraints=[
                {"type": "must_include", "value": "extra_check"},
            ],
        )
        graders = contract.to_graders()

        # Should have: json_schema, tool_call, latency, token_budget, contains, constraint
        assert len(graders) >= 5  # At least 5 different grader types

    def test_output_model_produces_structured_output_grader(self) -> None:
        contract = BehaviorContract(
            contract_id="test",
            version="1.0",
            output_model="eval_kit.core.transcript.ToolCall",
        )
        graders = contract.to_graders()

        struct_graders = [g for g, p in graders if "structured_output" in g.grader_id]
        assert len(struct_graders) == 1
        policies = [p for g, p in graders if "structured_output" in g.grader_id]
        assert policies[0] == EvalPolicy.GATE

    def test_output_schema_and_model_produce_both_graders(self) -> None:
        contract = BehaviorContract(
            contract_id="test",
            version="1.0",
            output_schema={"type": "object"},
            output_model="eval_kit.core.transcript.ToolCall",
        )
        graders = contract.to_graders()

        grader_ids = [g.grader_id for g, _ in graders]
        assert "test.json_schema" in grader_ids
        assert "test.structured_output" in grader_ids

    def test_to_graders_returns_grader_policy_pairs(self) -> None:
        contract = BehaviorContract(
            contract_id="test",
            version="1.0",
            output_schema={"type": "object"},
        )
        graders = contract.to_graders()
        for item in graders:
            assert isinstance(item, tuple)
            assert len(item) == 2
            g, p = item
            assert isinstance(g, Grader)
            assert isinstance(p, EvalPolicy)


class TestBehaviorContractIO:
    """Tests for loading/saving contracts."""

    def test_save_and_load_json(self) -> None:
        contract = BehaviorContract(
            contract_id="test",
            version="1.0",
            output_schema={"type": "object"},
            max_tokens=1000,
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(contract.model_dump(mode="json"), f)
            path = Path(f.name)

        try:
            with open(path) as f:
                data = json.load(f)
            restored = BehaviorContract.model_validate(data)
            assert restored.contract_id == "test"
            assert restored.max_tokens == 1000
        finally:
            path.unlink()
