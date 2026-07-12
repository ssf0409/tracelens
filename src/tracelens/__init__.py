"""TraceLens: friendly evaluation and regression testing for AI agents.

TraceLens turns agent runs into inspectable traces, graded outcomes,
baseline comparisons, and CI-ready reliability signals.

Top-level imports are the **stable public API**. Everything exposed in
``__all__`` is covered by the stability policy:

- Semver-style: breaking changes bump the major version.
- Submodule paths (e.g. ``tracelens.core.task.TaskExpectation``) are *not*
  covered — they may move. Import from the package root for anything you
  depend on long-term.

The framework is organised around five concerns:

- **Core models** — Task, Trial, Transcript, Outcome.
- **Reproducibility** — DecisionSpec captures the full agent configuration
  so runs are reproducible and regressions are attributable.
- **Graders** — CodeGrader / LLMGrader abstractions plus a library of
  built-in graders (schema, budget, tool-use, event-chain, ...).
- **Execution** — AgentAdapter + EvaluationRunner drive parallel trials;
  HTTPAPIAdapter handles remote agents over HTTP.
- **Regression detection** — BaselineManager + RegressionDetector gate
  CI with severity-aware, statistically-sound comparisons.

Optional dependencies:

- ``[llm]`` — ``openai``, ``anthropic`` (convenience bundle for subclassing LLMProvider with common vendor SDKs).
- ``[http]`` — ``httpx`` (for HTTPAPIAdapter).
- ``[dev]`` — test + lint + type-check tooling.

See the README for a 10-minute quickstart.
"""

# Imports are grouped alphabetically by module path (ruff-enforced).
# The topical grouping lives in `__all__` below, which is the readable
# index of the public API.

from tracelens._version import __version__  # noqa: F401  (re-exported)
from tracelens.baselines.comparison import (
    MetricRegression,
    RegressionDetector,
    RegressionReport,
    RegressionSeverity,
)
from tracelens.baselines.manager import (
    BaselineManager,
    BaselineType,
    MetricBaseline,
    PromotionPolicy,
    TaskBaseline,
)
from tracelens.contracts.contract import BehaviorContract
from tracelens.core.decision_spec import (
    AgentSpec,
    DecisionSpec,
    EnvironmentSpec,
    InfraConfig,
    ModelConfig,
    PromptSpec,
    ToolSpec,
)
from tracelens.core.grader import (
    CodeGrader,
    CompositeGrader,
    EvalPolicy,
    Grader,
    GraderConfig,
    GraderRole,  # kept for back-compat; prefer EvalPolicy for new code.
    GraderType,
    LLMGrader,
)
from tracelens.core.outcome import GradeLevel, Outcome
from tracelens.core.task import (
    EvalSet,
    JSONTaskLoader,
    Task,
    TaskLoader,
)
from tracelens.core.transcript import (
    StreamingEvent,
    StreamingEventType,
    Transcript,
    TranscriptStep,
)
from tracelens.core.trial import InfraError, Trial, TrialBatch, TrialStatus
from tracelens.execution.agent_adapter import AgentAdapter, SimpleAdapter
from tracelens.execution.http_adapter import (
    AuthConfig,
    AuthScheme,
    HTTPAdapterConfig,
    HTTPAPIAdapter,
    RetryConfig,
)
from tracelens.execution.runner import (
    DEFAULT_INFRA_EXCEPTION_TYPES,
    EvaluationRunner,
    RunnerConfig,
)
from tracelens.graders.event_chain import (
    EventChainConfig,
    EventChainVerifier,
    EventExpectation,
    EventMatchType,
    OrderingMode,
)

# LLM provider abstraction. TraceLens defines the LLMProvider ABC plus
# InMemoryProvider for testing; real provider calls (OpenAI, Anthropic,
# LiteLLM, Bedrock, whatever) are the user's responsibility via a
# straightforward subclass. See the factory module docstring for the
# canonical pattern.
from tracelens.llm.factory import create_provider
from tracelens.llm.provider import InMemoryProvider, LLMProvider
from tracelens.metrics.budgets import (
    LatencyGrader,
    TokenBudgetGrader,
    ToolCallGrader,
    TraceConsistencyGrader,
)
from tracelens.metrics.validators import (
    ConstraintGrader,
    ContainsGrader,
    JsonSchemaGrader,
    RegexMatchGrader,
    StructuredOutputGrader,
)
from tracelens.reporting.generator import ReportData, ReportGenerator, TaskSummary
from tracelens.statistics.consistency import (
    ConsistencyAnalyzer,
    pass_to_k,
    pass_to_k_estimator,
)
from tracelens.statistics.inference import (
    ComparisonResult,
    MetricEstimate,
    bootstrap_ci,
    compare_metrics,
    compare_to_baseline_summary,
    estimate_metric,
)
from tracelens.statistics.latency import (
    AggregateLatencyMetrics,
    LatencyAnalyzer,
    LatencyMetrics,
)
from tracelens.statistics.pass_at_k import (
    PassAtKAnalyzer,
    pass_at_k,
    pass_at_k_estimator,
)

__all__ = [
    # --- Core models ---
    "Task",
    "TaskLoader",
    "JSONTaskLoader",
    "EvalSet",
    "Trial",
    "TrialBatch",
    "TrialStatus",
    "InfraError",
    "Transcript",
    "TranscriptStep",
    "StreamingEvent",
    "StreamingEventType",
    "Outcome",
    "GradeLevel",
    # --- Graders (ABCs + config) ---
    "Grader",
    "CodeGrader",
    "LLMGrader",
    "CompositeGrader",
    "GraderType",
    "GraderConfig",
    "EvalPolicy",
    "GraderRole",  # deprecated; use EvalPolicy.
    # --- Built-in graders ---
    "EventChainVerifier",
    "EventExpectation",
    "EventChainConfig",
    "EventMatchType",
    "OrderingMode",
    "JsonSchemaGrader",
    "StructuredOutputGrader",
    "ContainsGrader",
    "RegexMatchGrader",
    "ConstraintGrader",
    "LatencyGrader",
    "TokenBudgetGrader",
    "ToolCallGrader",
    "TraceConsistencyGrader",
    # --- Contracts ---
    "BehaviorContract",
    # --- Reproducibility ---
    "DecisionSpec",
    "ModelConfig",
    "PromptSpec",
    "ToolSpec",
    "AgentSpec",
    "EnvironmentSpec",
    "InfraConfig",
    # --- Execution ---
    "AgentAdapter",
    "SimpleAdapter",
    "HTTPAPIAdapter",
    "HTTPAdapterConfig",
    "AuthConfig",
    "AuthScheme",
    "RetryConfig",
    "EvaluationRunner",
    "RunnerConfig",
    "DEFAULT_INFRA_EXCEPTION_TYPES",
    # --- Baselines & regression detection ---
    "BaselineManager",
    "BaselineType",
    "PromotionPolicy",
    "TaskBaseline",
    "MetricBaseline",
    "RegressionDetector",
    "RegressionReport",
    "RegressionSeverity",
    "MetricRegression",
    # --- Reporting ---
    "ReportGenerator",
    "ReportData",
    "TaskSummary",
    # --- Statistics ---
    "pass_at_k",
    "pass_at_k_estimator",
    "PassAtKAnalyzer",
    "pass_to_k",
    "pass_to_k_estimator",
    "ConsistencyAnalyzer",
    "bootstrap_ci",
    "estimate_metric",
    "compare_metrics",
    "compare_to_baseline_summary",
    "MetricEstimate",
    "ComparisonResult",
    "LatencyMetrics",
    "AggregateLatencyMetrics",
    "LatencyAnalyzer",
    # --- LLM provider abstraction ---
    "LLMProvider",
    "InMemoryProvider",
    "create_provider",
]
