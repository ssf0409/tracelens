"""Tests for DecisionSpec and related models."""

from datetime import UTC, datetime

from tracelens.core.decision_spec import (
    AgentSpec,
    DecisionSpec,
    EnvironmentSpec,
    InfraConfig,
    ModelConfig,
    PromptSpec,
    ToolSpec,
)


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_creation_minimal(self):
        """Test creating a ModelConfig with minimal fields."""
        config = ModelConfig(
            provider="anthropic",
            model_id="claude-3-opus-20240229",
        )
        assert config.provider == "anthropic"
        assert config.model_id == "claude-3-opus-20240229"
        assert config.temperature is None

    def test_creation_full(self):
        """Test creating a ModelConfig with all fields."""
        config = ModelConfig(
            provider="openai",
            model_id="gpt-4-turbo",
            model_version="2024-01-01",
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            max_tokens=4096,
            seed=42,
            stop_sequences=["END", "STOP"],
            extra_params={"presence_penalty": 0.5},
        )
        assert config.provider == "openai"
        assert config.temperature == 0.7
        assert config.seed == 42
        assert config.extra_params == {"presence_penalty": 0.5}

    def test_to_hash_dict(self):
        """Test hash dict generation."""
        config = ModelConfig(
            provider="anthropic",
            model_id="claude-3-opus",
            temperature=0.7,
        )
        hash_dict = config.to_hash_dict()
        assert hash_dict["provider"] == "anthropic"
        assert hash_dict["model_id"] == "claude-3-opus"
        assert hash_dict["temperature"] == 0.7


class TestPromptSpec:
    """Tests for PromptSpec."""

    def test_from_prompts(self):
        """Test creating PromptSpec from actual prompts."""
        system = "You are a helpful assistant."
        template = "Given {context}, do {task}."

        spec = PromptSpec.from_prompts(
            system_prompt=system,
            prompt_template=template,
            prompt_version="v1.0",
        )

        assert spec.system_prompt_hash is not None
        assert spec.prompt_template_hash is not None
        assert spec.prompt_version == "v1.0"
        # Full prompts not stored by default
        assert spec.system_prompt is None
        assert spec.prompt_template is None

    def test_from_prompts_with_storage(self):
        """Test creating PromptSpec with full prompt storage."""
        system = "You are a helpful assistant."

        spec = PromptSpec.from_prompts(
            system_prompt=system,
            store_full_prompts=True,
        )

        assert spec.system_prompt == system
        assert spec.system_prompt_hash is not None

    def test_hash_deterministic(self):
        """Test that hashing is deterministic."""
        prompt = "You are a helpful assistant."

        spec1 = PromptSpec.from_prompts(system_prompt=prompt)
        spec2 = PromptSpec.from_prompts(system_prompt=prompt)

        assert spec1.system_prompt_hash == spec2.system_prompt_hash


class TestToolSpec:
    """Tests for ToolSpec."""

    def test_creation(self):
        """Test creating a ToolSpec."""
        spec = ToolSpec(
            name="search",
            version="1.0.0",
            description_hash="abc123",
            schema_hash="def456",
        )
        assert spec.name == "search"
        assert spec.version == "1.0.0"

    def test_to_hash_dict(self):
        """Test hash dict generation."""
        spec = ToolSpec(name="calculator", version="2.0")
        hash_dict = spec.to_hash_dict()
        assert hash_dict["name"] == "calculator"
        assert hash_dict["version"] == "2.0"


class TestAgentSpec:
    """Tests for AgentSpec."""

    def test_creation(self):
        """Test creating an AgentSpec."""
        spec = AgentSpec(
            agent_name="goal_decomposition",
            agent_version="1.0.0",
            agent_graph_hash="graph123",
            config_hash="config456",
        )
        assert spec.agent_name == "goal_decomposition"
        assert spec.agent_version == "1.0.0"


class TestEnvironmentSpec:
    """Tests for EnvironmentSpec."""

    def test_creation(self):
        """Test creating an EnvironmentSpec."""
        spec = EnvironmentSpec(
            git_commit="abc123def456",
            git_branch="main",
            build_id="build-123",
            runner_version="0.1.0",
            framework_version="0.1.0",
            python_version="3.12.0",
        )
        assert spec.git_commit == "abc123def456"
        assert spec.build_id == "build-123"


class TestDecisionSpec:
    """Tests for DecisionSpec."""

    def test_creation_minimal(self):
        """Test creating a DecisionSpec with minimal fields."""
        spec = DecisionSpec()
        assert spec.model is None
        assert spec.fingerprint is not None
        assert len(spec.fingerprint) == 64  # SHA-256 hex

    def test_creation_full(self):
        """Test creating a DecisionSpec with all fields."""
        spec = DecisionSpec(
            model=ModelConfig(
                provider="anthropic",
                model_id="claude-3-opus",
                temperature=0.7,
            ),
            prompts=PromptSpec.from_prompts(
                system_prompt="You are helpful.",
                prompt_version="v1",
            ),
            tools=[
                ToolSpec(name="search", version="1.0"),
                ToolSpec(name="calculator", version="2.0"),
            ],
            agent=AgentSpec(
                agent_name="goal_decomposition",
                agent_version="1.0.0",
            ),
            environment=EnvironmentSpec(
                git_commit="abc123",
                framework_version="0.1.0",
            ),
            global_seed=42,
            extra={"debug": True},
        )

        assert spec.model.provider == "anthropic"
        assert len(spec.tools) == 2
        assert spec.global_seed == 42
        assert spec.fingerprint is not None

    def test_fingerprint_deterministic(self):
        """Test that fingerprint is deterministic for same inputs."""
        def create_spec():
            return DecisionSpec(
                model=ModelConfig(
                    provider="anthropic",
                    model_id="claude-3-opus",
                    temperature=0.7,
                ),
                tools=[ToolSpec(name="search")],
                global_seed=42,
            )

        spec1 = create_spec()
        spec2 = create_spec()

        assert spec1.fingerprint == spec2.fingerprint

    def test_fingerprint_differs_on_change(self):
        """Test that fingerprint changes when inputs change."""
        spec1 = DecisionSpec(
            model=ModelConfig(
                provider="anthropic",
                model_id="claude-3-opus",
                temperature=0.7,
            ),
        )
        spec2 = DecisionSpec(
            model=ModelConfig(
                provider="anthropic",
                model_id="claude-3-opus",
                temperature=0.8,  # Different temperature
            ),
        )

        assert spec1.fingerprint != spec2.fingerprint

    def test_fingerprint_short(self):
        """Test short fingerprint is first 12 characters."""
        spec = DecisionSpec(
            model=ModelConfig(provider="test", model_id="test"),
        )
        assert spec.fingerprint_short == spec.fingerprint[:12]
        assert len(spec.fingerprint_short) == 12

    def test_is_compatible_with_same_model(self):
        """Test compatibility check for same model."""
        spec1 = DecisionSpec(
            model=ModelConfig(provider="anthropic", model_id="claude-3-opus"),
            agent=AgentSpec(agent_name="agent1"),
        )
        spec2 = DecisionSpec(
            model=ModelConfig(provider="anthropic", model_id="claude-3-opus"),
            agent=AgentSpec(agent_name="agent1"),
            # Different prompts - still compatible
            prompts=PromptSpec(prompt_version="v2"),
        )

        assert spec1.is_compatible_with(spec2)

    def test_is_compatible_with_different_model(self):
        """Test compatibility check for different model."""
        spec1 = DecisionSpec(
            model=ModelConfig(provider="anthropic", model_id="claude-3-opus"),
        )
        spec2 = DecisionSpec(
            model=ModelConfig(provider="openai", model_id="gpt-4"),
        )

        assert not spec1.is_compatible_with(spec2)

    def test_diff_empty(self):
        """Test diff with identical specs."""
        spec = DecisionSpec(
            model=ModelConfig(provider="anthropic", model_id="claude-3"),
        )
        diff = spec.diff(spec)
        assert diff == {}

    def test_diff_with_changes(self):
        """Test diff with changed specs."""
        spec1 = DecisionSpec(
            model=ModelConfig(
                provider="anthropic",
                model_id="claude-3",
                temperature=0.7,
            ),
        )
        spec2 = DecisionSpec(
            model=ModelConfig(
                provider="anthropic",
                model_id="claude-3",
                temperature=0.8,
            ),
        )

        diff = spec1.diff(spec2)
        assert "model" in diff
        # The diff contains the full model dict, temperature is different
        model_diff = diff["model"]
        assert model_diff[0]["temperature"] == 0.7
        assert model_diff[1]["temperature"] == 0.8

    def test_to_summary(self):
        """Test summary generation."""
        spec = DecisionSpec(
            model=ModelConfig(
                provider="anthropic",
                model_id="claude-3-opus",
                temperature=0.7,
            ),
            agent=AgentSpec(
                agent_name="goal_decomposition",
                agent_version="1.0.0",
            ),
            tools=[
                ToolSpec(name="search"),
                ToolSpec(name="calculator"),
            ],
            environment=EnvironmentSpec(git_commit="abc123def456"),
        )

        summary = spec.to_summary()

        assert "fingerprint" in summary
        assert summary["model"] == "anthropic/claude-3-opus"
        assert summary["temperature"] == 0.7
        assert summary["agent"] == "goal_decomposition"
        assert summary["agent_version"] == "1.0.0"
        assert summary["tools"] == ["search", "calculator"]
        assert summary["git_commit"] == "abc123d"  # First 7 chars

    def test_tools_sorted_for_fingerprint(self):
        """Test that tools are sorted by name for consistent fingerprinting."""
        spec1 = DecisionSpec(
            tools=[
                ToolSpec(name="calculator"),
                ToolSpec(name="search"),
            ],
        )
        spec2 = DecisionSpec(
            tools=[
                ToolSpec(name="search"),
                ToolSpec(name="calculator"),
            ],
        )

        # Fingerprints should be same regardless of tool order
        assert spec1.fingerprint == spec2.fingerprint


class TestDecisionSpecIntegration:
    """Integration tests for DecisionSpec with Transcript."""

    def test_transcript_with_decision_spec(self):
        """Test that DecisionSpec works with Transcript."""
        from tracelens.core.transcript import Transcript

        spec = DecisionSpec(
            model=ModelConfig(
                provider="anthropic",
                model_id="claude-3-opus",
                temperature=0.7,
            ),
            agent=AgentSpec(agent_name="test_agent"),
        )

        transcript = Transcript(
            task_id="test-task",
            agent_name="test_agent",
            decision_spec=spec,
        )

        assert transcript.decision_spec is not None
        assert transcript.decision_spec.fingerprint == spec.fingerprint

        # Check summary includes fingerprint
        summary = transcript.to_summary()
        assert "fingerprint" in summary
        assert summary["fingerprint"] == spec.fingerprint_short

    def test_trial_fingerprint_property(self):
        """Test that Trial can access fingerprint from transcript."""
        from tracelens.core.transcript import Transcript
        from tracelens.core.trial import Trial

        spec = DecisionSpec(
            model=ModelConfig(provider="test", model_id="test-model"),
        )

        transcript = Transcript(
            task_id="test-task",
            decision_spec=spec,
        )

        trial = Trial(
            task_id="test-task",
            transcript=transcript,
        )

        assert trial.fingerprint == spec.fingerprint
        assert trial.fingerprint_short == spec.fingerprint_short

    def test_trial_fingerprint_none_without_spec(self):
        """Test that Trial fingerprint is None without decision_spec."""
        from tracelens.core.transcript import Transcript
        from tracelens.core.trial import Trial

        transcript = Transcript(task_id="test-task")
        trial = Trial(task_id="test-task", transcript=transcript)

        assert trial.fingerprint is None


class TestInfraConfig:
    """Tests for InfraConfig — the infrastructure-noise reproducibility spec."""

    def test_creation_minimal(self):
        """InfraConfig can be created with no fields (all optional)."""
        infra = InfraConfig()
        assert infra.cpu_guaranteed is None
        assert infra.memory_hard_limit_mb is None
        assert infra.runtime_platform is None

    def test_creation_with_split_limits(self):
        """The guaranteed/hard-limit split (Anthropic's core recommendation)
        is a first-class concept: both are stored, not a single pinned value."""
        infra = InfraConfig(
            cpu_guaranteed=1.0,
            cpu_hard_limit=3.0,
            memory_guaranteed_mb=512,
            memory_hard_limit_mb=1536,
            time_budget_seconds=600.0,
            concurrency_level=8,
            runtime_platform="kubernetes",
            sandbox_provider="gke",
            harness_version="tracelens-0.1.0",
        )
        assert infra.cpu_guaranteed == 1.0
        assert infra.cpu_hard_limit == 3.0
        assert infra.memory_guaranteed_mb == 512
        assert infra.memory_hard_limit_mb == 1536
        assert infra.time_budget_seconds == 600.0
        assert infra.runtime_platform == "kubernetes"

    def test_to_hash_dict_excludes_observational_fields(self):
        """Hostname, container ID, and wall-clock start are present on the
        model (for trace audits) but MUST NOT be part of the hash — two
        runs with the same config on different hosts must collide."""
        infra = InfraConfig(
            cpu_hard_limit=2.0,
            hostname="worker-42",
            container_id="pod-abc123",
            wall_clock_start_utc=datetime(2026, 4, 16, 14, 30, tzinfo=UTC),
        )
        hash_dict = infra.to_hash_dict()
        assert "hostname" not in hash_dict
        assert "container_id" not in hash_dict
        assert "wall_clock_start_utc" not in hash_dict
        assert hash_dict["cpu_hard_limit"] == 2.0

    def test_identical_configs_on_different_hosts_hash_equal(self):
        """Two runs with the same resource config but different hosts
        should produce the same hash dict (i.e., same fingerprint)."""
        a = InfraConfig(
            cpu_hard_limit=3.0, memory_hard_limit_mb=2048,
            hostname="host-a", container_id="pod-111",
        )
        b = InfraConfig(
            cpu_hard_limit=3.0, memory_hard_limit_mb=2048,
            hostname="host-b", container_id="pod-222",
        )
        assert a.to_hash_dict() == b.to_hash_dict()

    def test_different_cpu_limits_hash_differently(self):
        """Different resource budgets are a different experiment —
        the hash must reflect that, so baselines don't get compared
        across configs that shift scores by several percentage points."""
        a = InfraConfig(cpu_hard_limit=1.0)
        b = InfraConfig(cpu_hard_limit=3.0)
        assert a.to_hash_dict() != b.to_hash_dict()

    def test_different_memory_limits_hash_differently(self):
        """Memory limits fall in the same "experimental variable" bucket
        as CPU — Anthropic saw the strongest effect from memory changes."""
        a = InfraConfig(memory_hard_limit_mb=1024)
        b = InfraConfig(memory_hard_limit_mb=4096)
        assert a.to_hash_dict() != b.to_hash_dict()

    def test_sandbox_provider_affects_hash(self):
        """Different sandboxing providers enforce resource limits with
        different strictness — recording the provider is part of the
        experimental variable per Anthropic's recommendation."""
        a = InfraConfig(cpu_hard_limit=3.0, sandbox_provider="gke")
        b = InfraConfig(cpu_hard_limit=3.0, sandbox_provider="terminal-bench-sandbox")
        assert a.to_hash_dict() != b.to_hash_dict()


class TestDecisionSpecWithInfra:
    """Tests for DecisionSpec integration with InfraConfig."""

    def test_fingerprint_stable_when_infra_omitted(self):
        """Back-compat: specs without InfraConfig must produce the same
        fingerprint as before InfraConfig was introduced. We enforce this
        by only including 'infra' in the hash dict when it's non-None."""
        spec = DecisionSpec(
            model=ModelConfig(provider="anthropic", model_id="claude-3-opus"),
        )
        hash_dict = spec._to_hash_dict()
        # The key must be absent (not present with a null value), otherwise
        # any previously-stored baseline fingerprint would drift.
        assert "infra" not in hash_dict

    def test_fingerprint_changes_when_infra_added(self):
        """Adding an InfraConfig to an existing spec changes the
        fingerprint — infra is a first-class experimental variable."""
        model = ModelConfig(provider="anthropic", model_id="claude-3-opus")
        without = DecisionSpec(model=model)
        with_infra = DecisionSpec(model=model, infra=InfraConfig(cpu_hard_limit=3.0))
        assert without.fingerprint != with_infra.fingerprint

    def test_fingerprint_differs_for_different_infra(self):
        """Two specs identical except for InfraConfig must have
        different fingerprints — this is the whole point."""
        model = ModelConfig(provider="anthropic", model_id="claude-3-opus")
        tight = DecisionSpec(model=model, infra=InfraConfig(cpu_hard_limit=1.0))
        loose = DecisionSpec(model=model, infra=InfraConfig(cpu_hard_limit=5.0))
        assert tight.fingerprint != loose.fingerprint

    def test_fingerprint_same_for_observational_differences_only(self):
        """Two runs of the same experiment on different machines /
        at different times must collide to the same fingerprint."""
        model = ModelConfig(provider="anthropic", model_id="claude-3-opus")
        shared_infra_kwargs = dict(cpu_hard_limit=3.0, memory_hard_limit_mb=2048)
        run_a = DecisionSpec(
            model=model,
            infra=InfraConfig(
                **shared_infra_kwargs,
                hostname="runner-1",
                wall_clock_start_utc=datetime(2026, 4, 16, 9, 0, tzinfo=UTC),
            ),
        )
        run_b = DecisionSpec(
            model=model,
            infra=InfraConfig(
                **shared_infra_kwargs,
                hostname="runner-2",
                wall_clock_start_utc=datetime(2026, 4, 16, 21, 0, tzinfo=UTC),
            ),
        )
        assert run_a.fingerprint == run_b.fingerprint

    def test_diff_surfaces_infra_changes(self):
        """DecisionSpec.diff() should highlight infra drift between
        a baseline and a current run."""
        baseline = DecisionSpec(
            infra=InfraConfig(cpu_hard_limit=1.0, memory_hard_limit_mb=1024),
        )
        current = DecisionSpec(
            infra=InfraConfig(cpu_hard_limit=3.0, memory_hard_limit_mb=1024),
        )
        diffs = baseline.diff(current)
        assert "infra" in diffs
        baseline_infra, current_infra = diffs["infra"]
        assert baseline_infra["cpu_hard_limit"] == 1.0
        assert current_infra["cpu_hard_limit"] == 3.0

    def test_summary_exposes_infra_when_present(self):
        """to_summary() should surface key infra fields so reports can
        show the experimental configuration at a glance."""
        spec = DecisionSpec(
            model=ModelConfig(provider="anthropic", model_id="claude-3-opus"),
            infra=InfraConfig(
                cpu_guaranteed=1.0,
                cpu_hard_limit=3.0,
                memory_hard_limit_mb=2048,
                time_budget_seconds=600.0,
                runtime_platform="kubernetes",
            ),
        )
        summary = spec.to_summary()
        assert "infra" in summary
        assert summary["infra"]["platform"] == "kubernetes"
        # Guaranteed/hard-limit pair is rendered as "floor/ceiling".
        assert summary["infra"]["cpu"] == "1.0/3.0"
        assert summary["infra"]["time_budget_s"] == 600.0

    def test_summary_omits_infra_when_absent(self):
        """When no InfraConfig is attached, the 'infra' key must not
        appear in the summary — keeps back-compat for existing callers."""
        spec = DecisionSpec(
            model=ModelConfig(provider="anthropic", model_id="claude-3-opus"),
        )
        summary = spec.to_summary()
        assert "infra" not in summary
