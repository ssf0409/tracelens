# TraceLens vs Adjacent Tools

TraceLens is for local, inspectable, CI-ready agent evaluation. It is not trying
to replace hosted tracing dashboards, prompt playgrounds, or full observability
platforms.

Use this page to decide whether TraceLens is the right tool for the job in
front of you.

## Short Version

Choose TraceLens when you need to answer:

- Did this agent version regress against a known baseline?
- Is success reliable across repeated attempts, not just possible once?
- Did a prompt, model, tool, or infrastructure change alter outcomes?
- Can CI block lower-quality or unsafe behavior before it ships?
- Can reviewers inspect the transcripts and grader feedback behind the score?

Choose something else when your primary need is a hosted dashboard, live
production tracing, prompt iteration UI, or a large built-in metric suite for a
specific domain.

## What TraceLens Optimizes For

| Need | TraceLens fit |
|---|---|
| Local evals in a repo | Strong fit. Runs in Python, writes JSON/markdown/HTML artifacts. |
| CI regression gates | Strong fit. Baselines and severity thresholds are first-class. |
| Nondeterministic agents | Strong fit. pass@k and pass^k separate capability from reliability. |
| Reproducibility | Strong fit. `DecisionSpec` records model, prompt, tool, and infra fingerprints. |
| Harness reliability | Strong fit. Infra and grader errors are tracked separately from agent failures. |
| Human calibration | Good fit. Worksheet sampling and reconciliation are built in. |
| Hosted tracing UI | Not the goal. Export TraceLens outputs into a tracing system if needed. |
| Prompt playground | Not the goal. Pair TraceLens with your prompt workflow. |
| Provider matrix | Not the goal. Bring your own adapter or `LLMProvider` subclass. |
| Domain truth | Not the goal. Downstream projects own labels, graders, thresholds, and rollout policy. |

## Adjacent Categories

This is a capability comparison, not a vendor ranking.

### Hosted Tracing And Observability

Tools in this category are strongest when you need live traces, production
dashboards, request search, team collaboration, or online debugging.

TraceLens is different: it focuses on repeatable eval suites, local artifacts,
statistical reliability, and CI gating. It can consume or emit data that fits
an observability workflow, but it is not a hosted backend.

Pick TraceLens if the question is "should this agent change land?"

Pick a hosted tracing tool if the question is "what happened in production for
this user request?"

### Prompt Management And Experimentation

Prompt tools are strongest when you need prompt versioning, manual iteration,
playgrounds, or collaborative review of prompts and examples.

TraceLens is different: it treats prompt changes as one input to an eval run.
The important output is whether behavior changed enough to pass, fail, block,
or require human review.

Pick TraceLens if the prompt is already part of an agent system and you need a
regression gate.

Pick a prompt tool if the main workflow is interactive prompt design.

### Benchmark And Metric Suites

Benchmark suites are strongest when they provide canonical datasets, labels, or
metrics for a specific domain.

TraceLens is different: it supplies the harness, statistics, baselines, reports,
and calibration loop. Your project owns task data, labels, domain graders, and
promotion policy.

Pick TraceLens if your eval suite comes from your own failures, workflows, or
domain-specific acceptance criteria.

Pick a specialized benchmark suite when its dataset and metric exactly match
the decision you need to make.

### RAG Evaluation

RAG eval tools often focus on retrieval relevance, context precision, faithfulness,
and answer quality.

TraceLens can evaluate RAG systems, but it should usually do that through a
recipe: task data contains the query and expected evidence, an adapter runs the
system, and downstream graders score retrieval and answer quality. TraceLens
should not absorb a broad RAG metric suite into core until multiple real users
need the same abstraction.

## Decision Table

| If you need... | Start with... |
|---|---|
| A regression gate for an agent PR | TraceLens baselines + CI summary |
| Capability vs reliability over repeated runs | TraceLens pass@k + pass^k |
| A human calibration loop for LLM judges | TraceLens sample + reconcile |
| A live hosted trace explorer | Hosted observability/tracing tool |
| Prompt playground and prompt version UI | Prompt-management tool |
| Standardized domain leaderboard | Benchmark suite |
| RAG-specific metrics out of the box | RAG eval framework or a TraceLens recipe |

## How To Combine Tools

TraceLens does not need to be the only tool in the workflow:

1. Use your agent framework to run the system.
2. Use TraceLens to record transcripts, grade outcomes, compare baselines, and
   produce CI artifacts.
3. Export artifacts or traces to dashboards when humans need live exploration.

The boundary is simple: TraceLens should make the ship/no-ship signal
defensible. Other tools can make debugging, tracing, and iteration more
comfortable.
