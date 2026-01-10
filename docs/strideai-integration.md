# Phase 2A: StrideAI Integration Guide

This guide covers integrating the `agent-eval` framework with StrideAI's goal decomposition agent.

## Overview

StrideAI's goal decomposition agent takes user goals and context, then produces structured plans with phases, tasks, timelines, and resources. The evaluation focuses on **subjective quality** using LLM-as-judge graders.

## Directory Structure

Create this structure in your StrideAI project:

```
StrideAI/
├── eval/
│   ├── __init__.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── task_definition.py    # GoalDecompositionTask
│   ├── graders/
│   │   ├── __init__.py
│   │   ├── specificity_grader.py
│   │   ├── personalization_grader.py
│   │   ├── actionability_grader.py
│   │   └── constraint_grader.py
│   ├── data/
│   │   └── scenarios/            # 20+ test scenarios
│   │       ├── programming_beginner.json
│   │       ├── career_transition.json
│   │       ├── fitness_marathon.json
│   │       └── ...
│   ├── baselines/
│   │   └── baselines.json
│   ├── harness.py                # Evaluation orchestrator
│   └── human_eval/
│       └── app.py                # Streamlit calibration UI
└── pyproject.toml
```

## Step 1: Add Dependency

In `StrideAI/pyproject.toml`:

```toml
[project]
dependencies = [
    # ... existing dependencies
    "agent-eval[llm] @ git+https://github.com/ssf0409/agent-eval.git",
]
```

Install:
```bash
uv sync
```

## Step 2: Define Task Schema

Create `eval/schemas/task_definition.py`:

```python
"""Goal decomposition task schema."""

from typing import Any
from pydantic import BaseModel, Field

from agent_eval.core.task import Task, TaskExpectation


class UserContext(BaseModel):
    """User context for goal decomposition."""

    experience_level: str = Field(description="beginner, intermediate, expert")
    time_available: str = Field(description="hours per week")
    budget: str = Field(description="free, low, medium, high")
    constraints: list[str] = Field(default_factory=list)
    preferences: dict[str, Any] = Field(default_factory=dict)


class GoalDecompositionTask(Task):
    """Task for evaluating goal decomposition quality."""

    # Override input_data with typed structure
    goal: str = Field(description="The user's goal statement")
    user_context: UserContext = Field(description="User context and constraints")
    timeline: str = Field(description="Target timeline: short (2wk), medium (3mo), long (1yr)")

    @classmethod
    def from_json(cls, data: dict) -> "GoalDecompositionTask":
        """Create task from JSON data."""
        return cls(
            task_id=data["task_id"],
            name=data["name"],
            description=data.get("description", ""),
            goal=data["goal"],
            user_context=UserContext(**data["user_context"]),
            timeline=data["timeline"],
            input_data={
                "goal": data["goal"],
                "user_context": data["user_context"],
                "timeline": data["timeline"],
            },
            expectation=TaskExpectation(
                expected_metrics=data.get("expected_metrics", {}),
            ),
            category=data.get("category", "general"),
            tags=data.get("tags", []),
            difficulty=data.get("difficulty", "medium"),
        )
```

## Step 3: Create LLM Graders

### Specificity Grader

Create `eval/graders/specificity_grader.py`:

```python
"""Grader for task specificity - are tasks concrete and actionable?"""

from agent_eval.core.grader import LLMGrader
from agent_eval.core.transcript import Transcript
from agent_eval.core.task import Task


class SpecificityGrader(LLMGrader):
    """Grade how specific and concrete the generated tasks are."""

    grader_id = "specificity_grader"
    grader_version = "1.0.0"

    # LLM configuration
    model_name = "claude-sonnet-4-20250514"
    temperature = 0.0

    RUBRIC = """
You are evaluating the SPECIFICITY of a goal decomposition output.

## Scoring Criteria (1-10 scale):

### Excellent (9-10):
- Tasks use specific action verbs ("Complete Codecademy Python course" not "learn Python")
- Quantifiable targets included ("Build 3 practice projects", "Solve 50 LeetCode problems")
- Resources named specifically (actual course names, book titles, tools)
- Time estimates provided and realistic
- Success criteria are measurable

### Good (7-8):
- Most tasks are specific with clear actions
- Some quantifiable targets
- Resources mentioned but not always specific
- Time estimates present

### Acceptable (5-6):
- Mix of specific and vague tasks
- Some measurable elements
- Generic resource mentions ("find a tutorial")

### Poor (3-4):
- Mostly vague tasks ("learn about X", "understand Y")
- Few or no quantifiable targets
- No specific resources

### Fail (1-2):
- All tasks are vague and non-actionable
- No measurable success criteria
- Completely generic advice
"""

    def build_grading_prompt(self, transcript: Transcript, task: Task) -> str:
        output = transcript.final_output

        return f"""
{self.RUBRIC}

## Goal:
{task.input_data.get('goal', 'N/A')}

## User Context:
{task.input_data.get('user_context', {})}

## Agent Output:
{output}

## Instructions:
1. Analyze the output against the rubric above
2. Provide specific examples from the output that support your rating
3. Give a score from 1-10

Respond in this exact format:
SCORE: [number]
PASSED: [true/false] (true if score >= 7)
REASONING: [your detailed analysis with specific examples]
"""

    def parse_llm_response(
        self,
        response: str,
        task: Task
    ) -> tuple[bool, float, dict[str, float], str]:
        """Parse LLM response into structured result."""
        lines = response.strip().split("\n")

        score = 5.0
        passed = False
        reasoning = ""

        for line in lines:
            if line.startswith("SCORE:"):
                try:
                    score = float(line.split(":")[1].strip())
                except ValueError:
                    pass
            elif line.startswith("PASSED:"):
                passed = "true" in line.lower()
            elif line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()

        # Normalize score to 0-1 range
        normalized_score = score / 10.0

        return passed, normalized_score, {"specificity": normalized_score}, reasoning
```

### Personalization Grader

Create `eval/graders/personalization_grader.py`:

```python
"""Grader for personalization - does the plan match user context?"""

from agent_eval.core.grader import LLMGrader
from agent_eval.core.transcript import Transcript
from agent_eval.core.task import Task


class PersonalizationGrader(LLMGrader):
    """Grade how well the plan is personalized to user context."""

    grader_id = "personalization_grader"
    grader_version = "1.0.0"

    model_name = "claude-sonnet-4-20250514"
    temperature = 0.0

    RUBRIC = """
You are evaluating the PERSONALIZATION of a goal decomposition output.

## User Context Factors to Check:
- Experience level (beginner/intermediate/expert)
- Time availability (hours per week)
- Budget constraints
- Stated preferences and constraints
- Learning style hints

## Scoring Criteria (1-10 scale):

### Excellent (9-10):
- Plan explicitly acknowledges user's experience level
- Time estimates fit within stated availability
- Resources match budget constraints
- Constraints are all respected
- Recommendations feel tailored, not generic

### Good (7-8):
- Most context factors addressed
- Minor mismatches in time or budget
- Generally appropriate for experience level

### Acceptable (5-6):
- Some personalization evident
- Ignores some stated constraints
- Could apply to many users

### Poor (3-4):
- Minimal personalization
- Ignores key context (e.g., suggests paid resources for "free" budget)
- Experience level mismatch

### Fail (1-2):
- Completely generic plan
- Directly contradicts stated constraints
- Inappropriate for user's level
"""

    def build_grading_prompt(self, transcript: Transcript, task: Task) -> str:
        output = transcript.final_output
        context = task.input_data.get("user_context", {})

        return f"""
{self.RUBRIC}

## Goal:
{task.input_data.get('goal', 'N/A')}

## User Context:
- Experience Level: {context.get('experience_level', 'N/A')}
- Time Available: {context.get('time_available', 'N/A')}
- Budget: {context.get('budget', 'N/A')}
- Constraints: {context.get('constraints', [])}
- Preferences: {context.get('preferences', {})}

## Agent Output:
{output}

## Instructions:
1. Check how well the output respects each context factor
2. Note any mismatches or ignored constraints
3. Give a score from 1-10

Respond in this exact format:
SCORE: [number]
PASSED: [true/false] (true if score >= 7)
REASONING: [your detailed analysis]
"""

    def parse_llm_response(
        self,
        response: str,
        task: Task
    ) -> tuple[bool, float, dict[str, float], str]:
        lines = response.strip().split("\n")

        score = 5.0
        passed = False
        reasoning = ""

        for line in lines:
            if line.startswith("SCORE:"):
                try:
                    score = float(line.split(":")[1].strip())
                except ValueError:
                    pass
            elif line.startswith("PASSED:"):
                passed = "true" in line.lower()
            elif line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()

        normalized_score = score / 10.0

        return passed, normalized_score, {"personalization": normalized_score}, reasoning
```

### Actionability Grader

Create `eval/graders/actionability_grader.py`:

```python
"""Grader for actionability - can the user start immediately?"""

from agent_eval.core.grader import LLMGrader
from agent_eval.core.transcript import Transcript
from agent_eval.core.task import Task


class ActionabilityGrader(LLMGrader):
    """Grade how immediately actionable the first steps are."""

    grader_id = "actionability_grader"
    grader_version = "1.0.0"

    model_name = "claude-sonnet-4-20250514"
    temperature = 0.0

    RUBRIC = """
You are evaluating the ACTIONABILITY of a goal decomposition output.

## Key Question: Can the user start RIGHT NOW?

## Scoring Criteria (1-10 scale):

### Excellent (9-10):
- First task can be started within 5 minutes
- No prerequisites or setup needed for initial action
- Clear "do this first" instruction
- Links or specific resources provided for first step
- No decision paralysis (clear single path forward)

### Good (7-8):
- First steps are clear and actionable
- Minimal setup required
- Resources identified but may need searching

### Acceptable (5-6):
- Actionable but requires some preparation
- User needs to make decisions before starting
- Vague first steps

### Poor (3-4):
- Requires significant setup or research before starting
- Unclear where to begin
- Too many options without guidance

### Fail (1-2):
- Cannot determine where to start
- All tasks require other tasks first
- Completely theoretical with no concrete actions
"""

    def build_grading_prompt(self, transcript: Transcript, task: Task) -> str:
        output = transcript.final_output

        return f"""
{self.RUBRIC}

## Goal:
{task.input_data.get('goal', 'N/A')}

## Agent Output:
{output}

## Instructions:
1. Identify what the user would do FIRST based on this output
2. Evaluate if they could start in the next 5-10 minutes
3. Note any barriers to immediate action
4. Give a score from 1-10

Respond in this exact format:
SCORE: [number]
PASSED: [true/false] (true if score >= 7)
FIRST_ACTION: [what would the user do first]
REASONING: [your detailed analysis]
"""

    def parse_llm_response(
        self,
        response: str,
        task: Task
    ) -> tuple[bool, float, dict[str, float], str]:
        lines = response.strip().split("\n")

        score = 5.0
        passed = False
        reasoning = ""

        for line in lines:
            if line.startswith("SCORE:"):
                try:
                    score = float(line.split(":")[1].strip())
                except ValueError:
                    pass
            elif line.startswith("PASSED:"):
                passed = "true" in line.lower()
            elif line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()

        normalized_score = score / 10.0

        return passed, normalized_score, {"actionability": normalized_score}, reasoning
```

## Step 4: Create Test Scenarios

Create scenarios in `eval/data/scenarios/`. Example `programming_beginner.json`:

```json
{
  "task_id": "prog-beginner-001",
  "name": "Learn Python - Beginner",
  "description": "Complete beginner wants to learn Python for data science",
  "goal": "Learn Python programming well enough to do data analysis in 3 months",
  "user_context": {
    "experience_level": "beginner",
    "time_available": "10 hours per week",
    "budget": "free",
    "constraints": ["no prior programming experience", "prefer video content"],
    "preferences": {
      "learning_style": "visual",
      "goal_specificity": "data science focus"
    }
  },
  "timeline": "medium",
  "expected_metrics": {
    "specificity": 0.8,
    "personalization": 0.8,
    "actionability": 0.8
  },
  "category": "programming",
  "tags": ["python", "beginner", "data-science"],
  "difficulty": "medium"
}
```

Create 20+ scenarios covering:

| Domain | Short (2wk) | Medium (3mo) | Long (1yr) |
|--------|-------------|--------------|------------|
| Programming | Portfolio website | Python for data science | Full-stack developer |
| Career | Interview prep | UX design transition | Manager promotion |
| Fitness | Morning routine | 5K training | Marathon |
| Finance | Emergency fund | Investing basics | Pay off loans |
| Creative | Short story | Digital illustration | Self-publish novel |

Plus 5 edge cases:
- Multi-constraint (conflicting requirements)
- Ambiguous goal (vague user input)
- Expert level (advanced user)
- Urgent timeline (very short deadline)
- Conflicting goals (wants incompatible outcomes)

## Step 5: Create Evaluation Harness

Create `eval/harness.py`:

```python
"""Evaluation harness for StrideAI goal decomposition."""

import json
from pathlib import Path
from datetime import datetime

from agent_eval.core.task import EvalSet
from agent_eval.core.trial import Trial, TrialBatch, TrialStatus
from agent_eval.core.transcript import Transcript
from agent_eval.core.grader import CompositeGrader
from agent_eval.statistics.pass_at_k import PassAtKAnalyzer
from agent_eval.statistics.consistency import ConsistencyAnalyzer
from agent_eval.baselines.manager import BaselineManager
from agent_eval.baselines.comparison import RegressionDetector, RegressionSeverity

from eval.schemas.task_definition import GoalDecompositionTask
from eval.graders.specificity_grader import SpecificityGrader
from eval.graders.personalization_grader import PersonalizationGrader
from eval.graders.actionability_grader import ActionabilityGrader

# Import your agent
from app.agents.goal_decomposition.agent import GoalDecompositionAgent


class StrideAIEvaluator:
    """Orchestrates evaluation of the goal decomposition agent."""

    def __init__(
        self,
        scenarios_dir: Path = Path("eval/data/scenarios"),
        baselines_file: Path = Path("eval/baselines/baselines.json"),
        num_runs: int = 5,
    ):
        self.scenarios_dir = scenarios_dir
        self.baselines_file = baselines_file
        self.num_runs = num_runs

        # Initialize components
        self.agent = GoalDecompositionAgent()
        self.baseline_manager = BaselineManager(baselines_file)

        # Create composite grader
        self.grader = CompositeGrader(
            graders=[
                SpecificityGrader(),
                PersonalizationGrader(),
                ActionabilityGrader(),
            ],
            aggregation="mean",
        )

        # Statistics analyzers
        self.pass_at_k = PassAtKAnalyzer(k_values=[1, 3, 5])
        self.consistency = ConsistencyAnalyzer(k_values=[2, 3, 5])

    def load_tasks(self) -> EvalSet:
        """Load all test scenarios."""
        tasks = []
        for json_file in self.scenarios_dir.glob("*.json"):
            with open(json_file) as f:
                data = json.load(f)
                task = GoalDecompositionTask.from_json(data)
                tasks.append(task)

        return EvalSet(
            eval_set_id="strideai-goal-decomposition",
            name="StrideAI Goal Decomposition Evaluation",
            tasks=tasks,
        )

    async def run_single_trial(
        self,
        task: GoalDecompositionTask,
        run_index: int,
    ) -> Trial:
        """Run a single evaluation trial."""
        trial = Trial(
            trial_id=f"{task.task_id}-run-{run_index}",
            task_id=task.task_id,
            run_index=run_index,
            total_runs=self.num_runs,
            status=TrialStatus.RUNNING,
            started_at=datetime.utcnow(),
        )

        try:
            # Create transcript
            transcript = Transcript(
                transcript_id=f"transcript-{trial.trial_id}",
                task_id=task.task_id,
                agent_name="goal_decomposition",
                agent_version="1.0.0",
                started_at=datetime.utcnow(),
            )

            # Run agent
            result = await self.agent.run(
                goal=task.goal,
                user_context=task.user_context.model_dump(),
            )

            transcript.final_output = result
            transcript.completed_at = datetime.utcnow()
            trial.transcript = transcript

            # Grade the result
            outcome = await self.grader.grade(transcript, task)
            trial.add_outcome(outcome)

            trial.status = TrialStatus.COMPLETED
            trial.completed_at = datetime.utcnow()

        except Exception as e:
            trial.status = TrialStatus.FAILED
            trial.error_message = str(e)
            trial.completed_at = datetime.utcnow()

        return trial

    async def run_evaluation(
        self,
        categories: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> dict:
        """Run full evaluation suite."""
        eval_set = self.load_tasks()

        # Filter if specified
        if categories or tags:
            eval_set = eval_set.filter_tasks(categories=categories, tags=tags)

        batch = TrialBatch(
            batch_id=f"eval-{datetime.utcnow().isoformat()}",
            eval_set_id=eval_set.eval_set_id,
        )

        # Run trials
        for task in eval_set.tasks:
            for run_idx in range(self.num_runs):
                trial = await self.run_single_trial(task, run_idx)
                batch.add_trial(trial)

        # Compute statistics
        pass_results = batch.get_pass_results_by_task()

        capability = self.pass_at_k.analyze(pass_results)
        reliability = self.consistency.analyze(pass_results)

        # Check for regressions
        detector = RegressionDetector(min_delta_percent=5.0)
        regression_reports = {}

        for task_id, results in pass_results.items():
            baseline = self.baseline_manager.get_baseline(task_id)
            if baseline:
                # Get metrics from trials
                task_trials = batch.get_trials_for_task(task_id)
                current_metrics = [
                    trial.outcomes[0].metrics
                    for trial in task_trials
                    if trial.outcomes
                ]
                report = detector.compare(baseline, current_metrics)
                regression_reports[task_id] = report

        return {
            "batch": batch,
            "capability": capability,
            "reliability": reliability,
            "regression_reports": regression_reports,
            "summary": self._generate_summary(batch, capability, reliability),
        }

    def _generate_summary(self, batch, capability, reliability) -> dict:
        """Generate evaluation summary."""
        trials = batch.trials
        passed = sum(1 for t in trials if t.passed)
        failed = sum(1 for t in trials if not t.passed)

        return {
            "total_trials": len(trials),
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / len(trials) if trials else 0,
            "pass@1": capability.get("pass@1", 0),
            "pass@5": capability.get("pass@5", 0),
            "reliability_score": reliability.get("reliability_score", 0),
        }
```

## Step 6: Run Evaluation

```python
import asyncio
from eval.harness import StrideAIEvaluator

async def main():
    evaluator = StrideAIEvaluator(num_runs=5)

    # Run full evaluation
    results = await evaluator.run_evaluation()

    print(f"Pass Rate: {results['summary']['pass_rate']:.2%}")
    print(f"pass@1: {results['summary']['pass@1']:.2%}")
    print(f"pass@5: {results['summary']['pass@5']:.2%}")
    print(f"Reliability: {results['summary']['reliability_score']:.2%}")

    # Check regressions
    for task_id, report in results['regression_reports'].items():
        if report.has_regression:
            print(f"REGRESSION in {task_id}: {report.overall_severity}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Human Calibration (Weekly)

See [CI/CD Integration Guide](./ci-cd-integration.md) for setting up weekly calibration workflows.

## Success Criteria

1. **Mean quality score** ≥ 7/10 across all scenarios
2. **pass@5** ≥ 95% (high capability)
3. **pass^3** ≥ 80% (good reliability)
4. **LLM-human correlation** > 0.7 on all dimensions
5. **No regressions** blocking CI on severity ≥ MODERATE
