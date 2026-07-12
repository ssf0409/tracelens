# API Reference

Auto-generated from docstrings. Everything below is importable from the package
root (`from tracelens import ...`) and covered by the stability policy — submodule
paths may move, so import from the root for anything you depend on long-term.

!!! tip
    New here? Read [Core Concepts & Glossary](concepts.md) first for the mental
    model, then the [User Guide](user-guide.md) for guided usage. This page is
    the exhaustive index.

## Core models

::: tracelens.Task
::: tracelens.TaskLoader
::: tracelens.JSONTaskLoader
::: tracelens.JSONLTaskLoader
::: tracelens.CSVTaskLoader
::: tracelens.EvalSet
::: tracelens.Transcript
::: tracelens.TranscriptStep
::: tracelens.StreamingEvent
::: tracelens.StreamingEventType
::: tracelens.Outcome
::: tracelens.GradeLevel
::: tracelens.Trial
::: tracelens.TrialBatch
::: tracelens.TrialStatus
::: tracelens.InfraError

## Execution

::: tracelens.AgentAdapter
::: tracelens.SimpleAdapter
::: tracelens.HTTPAPIAdapter
::: tracelens.HTTPAdapterConfig
::: tracelens.AuthConfig
::: tracelens.AuthScheme
::: tracelens.RetryConfig
::: tracelens.EvaluationRunner
::: tracelens.RunnerConfig
::: tracelens.CheckpointError
::: tracelens.DEFAULT_INFRA_EXCEPTION_TYPES

## Graders — base classes

::: tracelens.Grader
::: tracelens.CodeGrader
::: tracelens.LLMGrader
::: tracelens.CompositeGrader
::: tracelens.GraderConfig
::: tracelens.GraderType
::: tracelens.EvalPolicy
::: tracelens.GraderRole
::: tracelens.BehaviorContract

## Graders — built-in library

See the [Grader Library](grader-library.md) guide for when to reach for each.

::: tracelens.JsonSchemaGrader
::: tracelens.RegexMatchGrader
::: tracelens.ContainsGrader
::: tracelens.ConstraintGrader
::: tracelens.StructuredOutputGrader
::: tracelens.LatencyGrader
::: tracelens.TokenBudgetGrader
::: tracelens.ToolCallGrader
::: tracelens.TraceConsistencyGrader
::: tracelens.EventChainVerifier
::: tracelens.EventChainConfig
::: tracelens.EventExpectation
::: tracelens.EventMatchType
::: tracelens.OrderingMode

## Statistics

See [pass@k vs pass^k](pass-at-k-vs-pass-hat-k.md) and
[Statistical Comparison](statistical-comparison.md) for the concepts.

::: tracelens.pass_at_k
::: tracelens.pass_at_k_estimator
::: tracelens.PassAtKAnalyzer
::: tracelens.pass_to_k
::: tracelens.pass_to_k_estimator
::: tracelens.ConsistencyAnalyzer
::: tracelens.bootstrap_ci
::: tracelens.estimate_metric
::: tracelens.compare_metrics
::: tracelens.compare_to_baseline_summary
::: tracelens.MetricEstimate
::: tracelens.ComparisonResult
::: tracelens.LatencyAnalyzer
::: tracelens.LatencyMetrics
::: tracelens.AggregateLatencyMetrics

## Baselines & regression detection

See the [Baseline Regression Tutorial](baseline-regression-tutorial.md).

::: tracelens.BaselineManager
::: tracelens.BaselineType
::: tracelens.PromotionPolicy
::: tracelens.MetricBaseline
::: tracelens.TaskBaseline
::: tracelens.RegressionDetector
::: tracelens.RegressionReport
::: tracelens.RegressionSeverity
::: tracelens.MetricRegression

## Reproducibility (DecisionSpec)

See [Reproducibility & DecisionSpec](reproducibility.md).

::: tracelens.DecisionSpec
::: tracelens.ModelConfig
::: tracelens.PromptSpec
::: tracelens.ToolSpec
::: tracelens.AgentSpec
::: tracelens.InfraConfig
::: tracelens.EnvironmentSpec

## LLM judge providers

::: tracelens.LLMProvider
::: tracelens.InMemoryProvider
::: tracelens.create_provider

## Reporting

::: tracelens.ReportGenerator
::: tracelens.ReportData
::: tracelens.TaskSummary
