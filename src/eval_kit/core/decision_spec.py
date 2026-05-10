"""Decision specification for reproducibility.

A DecisionSpec captures all parameters that could affect agent behavior,
enabling exact reproducibility of evaluation runs. The fingerprint is a
hash of all these parameters.

Research-grade evaluations require knowing:
- What model was used (provider + version)
- What prompts were used (hash of templates)
- What decoding parameters were used
- What version of the agent code was running
- What tools were available
"""

import hashlib
import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, computed_field


class ModelConfig(BaseModel):
    """Configuration for the LLM model used.

    Captures provider, model version, and all decoding parameters
    that could affect output.
    """

    provider: str = Field(
        description="Model provider (e.g., 'anthropic', 'openai', 'google')"
    )
    model_id: str = Field(
        description="Model identifier (e.g., 'claude-3-opus-20240229')"
    )
    model_version: str | None = Field(
        default=None,
        description="Specific version or snapshot if available",
    )

    # Decoding parameters
    temperature: float | None = Field(
        default=None,
        description="Temperature for sampling (0.0 = deterministic)",
    )
    top_p: float | None = Field(
        default=None,
        description="Nucleus sampling parameter",
    )
    top_k: int | None = Field(
        default=None,
        description="Top-k sampling parameter",
    )
    max_tokens: int | None = Field(
        default=None,
        description="Maximum tokens to generate",
    )
    seed: int | None = Field(
        default=None,
        description="Random seed for reproducibility (if supported)",
    )
    stop_sequences: list[str] | None = Field(
        default=None,
        description="Stop sequences for generation",
    )

    # Additional parameters
    extra_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Any additional model-specific parameters",
    )

    def to_hash_dict(self) -> dict[str, Any]:
        """Return dict of fields that affect output (for hashing)."""
        return {
            "provider": self.provider,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "stop_sequences": sorted(self.stop_sequences) if self.stop_sequences else None,
            "extra_params": self.extra_params,
        }


class PromptSpec(BaseModel):
    """Specification of prompts used in the agent.

    Stores hashes of prompt templates for traceability without
    storing the full prompts (which may be long or sensitive).
    """

    system_prompt_hash: str | None = Field(
        default=None,
        description="SHA-256 hash of the system prompt",
    )
    prompt_template_hash: str | None = Field(
        default=None,
        description="SHA-256 hash of the main prompt template",
    )
    prompt_version: str | None = Field(
        default=None,
        description="Version identifier for the prompt set",
    )

    # Optional: store actual prompts if not sensitive
    system_prompt: str | None = Field(
        default=None,
        description="Full system prompt (optional, for debugging)",
    )
    prompt_template: str | None = Field(
        default=None,
        description="Full prompt template (optional, for debugging)",
    )

    @classmethod
    def from_prompts(
        cls,
        system_prompt: str | None = None,
        prompt_template: str | None = None,
        prompt_version: str | None = None,
        store_full_prompts: bool = False,
    ) -> "PromptSpec":
        """Create PromptSpec from actual prompts.

        Args:
            system_prompt: The system prompt text
            prompt_template: The prompt template text
            prompt_version: Optional version identifier
            store_full_prompts: Whether to store full prompts (default False)
        """
        system_hash = None
        template_hash = None

        if system_prompt:
            system_hash = hashlib.sha256(system_prompt.encode()).hexdigest()
        if prompt_template:
            template_hash = hashlib.sha256(prompt_template.encode()).hexdigest()

        return cls(
            system_prompt_hash=system_hash,
            prompt_template_hash=template_hash,
            prompt_version=prompt_version,
            system_prompt=system_prompt if store_full_prompts else None,
            prompt_template=prompt_template if store_full_prompts else None,
        )

    def to_hash_dict(self) -> dict[str, Any]:
        """Return dict of fields for hashing."""
        return {
            "system_prompt_hash": self.system_prompt_hash,
            "prompt_template_hash": self.prompt_template_hash,
            "prompt_version": self.prompt_version,
        }


class ToolSpec(BaseModel):
    """Specification of a tool available to the agent.

    Captures tool identity and version for reproducibility.
    """

    name: str = Field(description="Tool name")
    version: str | None = Field(default=None, description="Tool version")
    description_hash: str | None = Field(
        default=None,
        description="Hash of tool description (affects LLM tool selection)",
    )
    schema_hash: str | None = Field(
        default=None,
        description="Hash of tool input/output schema",
    )

    def to_hash_dict(self) -> dict[str, Any]:
        """Return dict for hashing."""
        return {
            "name": self.name,
            "version": self.version,
            "description_hash": self.description_hash,
            "schema_hash": self.schema_hash,
        }


class AgentSpec(BaseModel):
    """Specification of the agent being evaluated.

    Captures agent identity, version, and graph structure.
    """

    agent_name: str = Field(description="Name of the agent")
    agent_version: str | None = Field(default=None, description="Agent version")
    agent_graph_hash: str | None = Field(
        default=None,
        description="Hash of agent graph structure (for multi-agent systems)",
    )
    config_hash: str | None = Field(
        default=None,
        description="Hash of agent configuration",
    )

    def to_hash_dict(self) -> dict[str, Any]:
        """Return dict for hashing."""
        return {
            "agent_name": self.agent_name,
            "agent_version": self.agent_version,
            "agent_graph_hash": self.agent_graph_hash,
            "config_hash": self.config_hash,
        }


class EnvironmentSpec(BaseModel):
    """Specification of the execution environment.

    Captures build/deployment information for traceability.
    """

    git_commit: str | None = Field(
        default=None,
        description="Git commit SHA of the codebase",
    )
    git_branch: str | None = Field(
        default=None,
        description="Git branch name",
    )
    build_id: str | None = Field(
        default=None,
        description="CI/CD build identifier",
    )
    runner_version: str | None = Field(
        default=None,
        description="Version of the evaluation runner",
    )
    framework_version: str | None = Field(
        default=None,
        description="Version of agent-eval framework",
    )
    python_version: str | None = Field(
        default=None,
        description="Python version",
    )

    def to_hash_dict(self) -> dict[str, Any]:
        """Return dict for hashing."""
        return {
            "git_commit": self.git_commit,
            "build_id": self.build_id,
            "runner_version": self.runner_version,
            "framework_version": self.framework_version,
        }


class InfraConfig(BaseModel):
    """Infrastructure / runtime-environment configuration.

    Agentic evals are end-to-end system tests: the runtime environment is
    part of the problem-solving process. Resource limits, time budgets, and
    concurrency all influence what strategies an agent can use, which means
    infrastructure configuration is a first-class experimental variable —
    not passive scaffolding.

    Anthropic's "Quantifying infrastructure noise in agentic coding evals"
    (Feb 2026) showed that infrastructure config alone can swing
    Terminal-Bench 2.0 scores by ~6 percentage points, often more than the
    leaderboard gap between frontier models. Their recommendations are
    baked into the shape of this spec:

    1. Record **both** a guaranteed allocation and a separate hard kill
       threshold, per task (see ``cpu_guaranteed`` / ``cpu_hard_limit``
       and ``memory_guaranteed_mb`` / ``memory_hard_limit_mb``). Pinning
       them to the same value leaves zero headroom for transient spikes
       and produces spurious infra failures.
    2. Capture the sandboxing provider, because enforcement semantics
       differ across runtimes (Kubernetes vs. Docker vs. Fly.io vs. bare
       containers).
    3. Keep observational fields (``hostname``, ``container_id``,
       ``wall_clock_start_utc``) out of the fingerprint so two runs with
       identical resource configs on different hosts collide to the same
       fingerprint.

    See: https://www.anthropic.com/engineering/infrastructure-noise
    """

    # --- Behavior-affecting fields (included in fingerprint) ---

    cpu_guaranteed: float | None = Field(
        default=None,
        description="Guaranteed CPU allocation in whole cores (e.g. 1.0 = 1 core).",
    )
    cpu_hard_limit: float | None = Field(
        default=None,
        description="CPU hard limit / kill threshold in whole cores.",
    )
    memory_guaranteed_mb: int | None = Field(
        default=None,
        description="Guaranteed memory allocation in MB.",
    )
    memory_hard_limit_mb: int | None = Field(
        default=None,
        description="Memory hard limit / OOM threshold in MB.",
    )
    time_budget_seconds: float | None = Field(
        default=None,
        description="Per-task wall-clock time budget in seconds.",
    )
    concurrency_level: int | None = Field(
        default=None,
        description="Max concurrent trials (affects resource contention).",
    )
    runtime_platform: str | None = Field(
        default=None,
        description=(
            "Container / runtime platform "
            "(e.g. 'kubernetes', 'docker', 'local', 'fly.io')."
        ),
    )
    sandbox_provider: str | None = Field(
        default=None,
        description=(
            "Sandboxing provider. Enforcement semantics "
            "(e.g. lenient vs. strict OOM handling) vary by provider."
        ),
    )
    harness_version: str | None = Field(
        default=None,
        description="Version of the eval harness (eval_kit or downstream wrapper).",
    )

    # --- Observational fields (NOT included in fingerprint) ---

    hostname: str | None = Field(
        default=None,
        description="Host running the eval. Observational; not in fingerprint.",
    )
    container_id: str | None = Field(
        default=None,
        description="Container / pod ID. Observational; not in fingerprint.",
    )
    wall_clock_start_utc: datetime | None = Field(
        default=None,
        description=(
            "UTC start time of the run. Useful for auditing "
            "time-of-day effects (e.g. upstream API latency spikes); "
            "not in fingerprint."
        ),
    )

    def to_hash_dict(self) -> dict[str, Any]:
        """Return dict of behavior-affecting fields for hashing.

        Observational fields (``hostname``, ``container_id``,
        ``wall_clock_start_utc``) are intentionally excluded: two runs
        with identical resource configs on different hosts should
        collide to the same fingerprint.
        """
        return {
            "cpu_guaranteed": self.cpu_guaranteed,
            "cpu_hard_limit": self.cpu_hard_limit,
            "memory_guaranteed_mb": self.memory_guaranteed_mb,
            "memory_hard_limit_mb": self.memory_hard_limit_mb,
            "time_budget_seconds": self.time_budget_seconds,
            "concurrency_level": self.concurrency_level,
            "runtime_platform": self.runtime_platform,
            "sandbox_provider": self.sandbox_provider,
            "harness_version": self.harness_version,
        }


class DecisionSpec(BaseModel):
    """Complete specification of all decision-affecting parameters.

    A DecisionSpec is an immutable fingerprint of everything that could
    affect agent behavior. Two runs with the same DecisionSpec fingerprint
    should produce statistically similar results (given the same task input).

    Example:
        spec = DecisionSpec(
            model=ModelConfig(
                provider="anthropic",
                model_id="claude-3-opus-20240229",
                temperature=0.7,
            ),
            prompts=PromptSpec.from_prompts(
                system_prompt="You are a helpful assistant...",
                prompt_template="Given {context}, do {task}...",
            ),
            tools=[
                ToolSpec(name="search", version="1.0"),
                ToolSpec(name="calculator", version="2.1"),
            ],
            agent=AgentSpec(
                agent_name="goal_decomposition",
                agent_version="1.0.0",
            ),
            environment=EnvironmentSpec(
                git_commit="abc123",
                framework_version="0.1.0",
            ),
        )
        print(spec.fingerprint)  # "a1b2c3d4..."
    """

    # Model configuration
    model: ModelConfig | None = Field(
        default=None,
        description="LLM model configuration",
    )

    # Prompt configuration
    prompts: PromptSpec | None = Field(
        default=None,
        description="Prompt templates and versions",
    )

    # Available tools
    tools: list[ToolSpec] = Field(
        default_factory=list,
        description="List of tools available to the agent",
    )

    # Agent configuration
    agent: AgentSpec | None = Field(
        default=None,
        description="Agent specification",
    )

    # Environment configuration
    environment: EnvironmentSpec | None = Field(
        default=None,
        description="Execution environment specification",
    )

    # Infrastructure / runtime configuration (resource limits, platform, etc.)
    infra: InfraConfig | None = Field(
        default=None,
        description=(
            "Runtime infrastructure configuration. Included in the "
            "fingerprint so runs with different resource budgets don't "
            "collide. See InfraConfig docstring for the distinction "
            "between behavior-affecting and observational fields."
        ),
    )

    # Global random seed
    global_seed: int | None = Field(
        default=None,
        description="Global random seed for reproducibility",
    )

    # Additional parameters that affect behavior
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Any additional parameters that affect agent behavior",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def fingerprint(self) -> str:
        """Compute SHA-256 fingerprint of the decision spec.

        This fingerprint uniquely identifies the configuration.
        Two runs with the same fingerprint should produce statistically
        similar results.
        """
        hash_data = self._to_hash_dict()
        # Serialize deterministically (sorted keys)
        serialized = json.dumps(hash_data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def fingerprint_short(self) -> str:
        """Short version of fingerprint (first 12 characters)."""
        return self.fingerprint[:12]

    def _to_hash_dict(self) -> dict[str, Any]:
        """Build dict for hashing."""
        result: dict[str, Any] = {
            "model": self.model.to_hash_dict() if self.model else None,
            "prompts": self.prompts.to_hash_dict() if self.prompts else None,
            "tools": [t.to_hash_dict() for t in sorted(self.tools, key=lambda t: t.name)],
            "agent": self.agent.to_hash_dict() if self.agent else None,
            "environment": self.environment.to_hash_dict() if self.environment else None,
            "global_seed": self.global_seed,
            "extra": self.extra,
        }
        # Only include ``infra`` in the hash when it's explicitly set so
        # fingerprints recorded before InfraConfig existed stay stable.
        # Adding the key unconditionally (with a null value for specs
        # without infra) would change every stored baseline's fingerprint.
        if self.infra is not None:
            result["infra"] = self.infra.to_hash_dict()
        return result

    def is_compatible_with(self, other: "DecisionSpec") -> bool:
        """Check if two specs are compatible for comparison.

        Two specs are compatible if they have the same model and agent,
        even if prompts or environment differ. This is useful for
        comparing prompt changes while keeping other factors constant.
        """
        if self.model and other.model:
            if self.model.provider != other.model.provider:
                return False
            if self.model.model_id != other.model.model_id:
                return False
        if self.agent and other.agent:
            if self.agent.agent_name != other.agent.agent_name:
                return False
        return True

    def diff(self, other: "DecisionSpec") -> dict[str, tuple[Any, Any]]:
        """Compare two specs and return differences.

        Returns dict mapping field names to (self_value, other_value) tuples
        for fields that differ.
        """
        diffs: dict[str, tuple[Any, Any]] = {}

        self_dict = self._to_hash_dict()
        other_dict = other._to_hash_dict()

        all_keys = set(self_dict.keys()) | set(other_dict.keys())
        for key in all_keys:
            self_val = self_dict.get(key)
            other_val = other_dict.get(key)
            if self_val != other_val:
                diffs[key] = (self_val, other_val)

        return diffs

    def to_summary(self) -> dict[str, Any]:
        """Create human-readable summary."""
        summary: dict[str, Any] = {
            "fingerprint": self.fingerprint_short,
        }

        if self.model:
            summary["model"] = f"{self.model.provider}/{self.model.model_id}"
            if self.model.temperature is not None:
                summary["temperature"] = self.model.temperature

        if self.agent:
            summary["agent"] = self.agent.agent_name
            if self.agent.agent_version:
                summary["agent_version"] = self.agent.agent_version

        if self.tools:
            summary["tools"] = [t.name for t in self.tools]

        if self.environment and self.environment.git_commit:
            summary["git_commit"] = self.environment.git_commit[:7]

        if self.infra:
            infra_summary: dict[str, Any] = {}
            if self.infra.runtime_platform:
                infra_summary["platform"] = self.infra.runtime_platform
            if self.infra.cpu_hard_limit is not None:
                infra_summary["cpu"] = (
                    f"{self.infra.cpu_guaranteed}/{self.infra.cpu_hard_limit}"
                    if self.infra.cpu_guaranteed is not None
                    else str(self.infra.cpu_hard_limit)
                )
            if self.infra.memory_hard_limit_mb is not None:
                infra_summary["memory_mb"] = (
                    f"{self.infra.memory_guaranteed_mb}/{self.infra.memory_hard_limit_mb}"
                    if self.infra.memory_guaranteed_mb is not None
                    else str(self.infra.memory_hard_limit_mb)
                )
            if self.infra.time_budget_seconds is not None:
                infra_summary["time_budget_s"] = self.infra.time_budget_seconds
            if self.infra.concurrency_level is not None:
                infra_summary["concurrency"] = self.infra.concurrency_level
            if infra_summary:
                summary["infra"] = infra_summary

        return summary
