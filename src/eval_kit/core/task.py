"""Task definitions for evaluation.

Tasks are the fundamental unit of evaluation - they define what to test
and what outcomes are expected.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any
import json
import uuid

from pydantic import BaseModel, Field


class TaskExpectation(BaseModel):
    """Expected outcomes for validation.

    These are optional hints that graders can use to validate outputs.
    Not all graders require expectations - LLM graders often work without them.
    """

    expected_output: Any | None = None
    expected_tool_calls: list[str] | None = None
    validation_hints: dict[str, Any] = Field(default_factory=dict)

    # For code-based graders
    expected_metrics: dict[str, float] | None = None
    metric_thresholds: dict[str, float] | None = None


class Task(BaseModel):
    """Represents a single evaluation task/test case.

    A Task defines:
    - Input data to feed to the agent
    - Optional expected outputs for validation
    - Metadata for filtering and categorization
    - Configuration for execution

    Example:
        task = Task(
            name="Portfolio website decomposition",
            input_data={
                "goal": "Build a personal portfolio website",
                "user_context": {"experience": "beginner", "hours_per_week": 15}
            },
            category="programming",
            tags=["web", "beginner"],
        )
    """

    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str | None = None

    # Input data - flexible for different agent types
    input_data: dict[str, Any]

    # Expected outcomes (optional - not all graders need this)
    expectation: TaskExpectation | None = None

    # Metadata for filtering and categorization
    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)

    # Difficulty/complexity hints for sampling
    difficulty: str | None = None  # "easy", "medium", "hard"
    category: str | None = None

    # Execution configuration
    timeout_seconds: float = 300.0
    max_retries: int = 1

    def matches_filter(
        self,
        tags: list[str] | None = None,
        categories: list[str] | None = None,
        difficulties: list[str] | None = None,
    ) -> bool:
        """Check if task matches filter criteria."""
        if tags and not any(t in self.tags for t in tags):
            return False
        if categories and self.category not in categories:
            return False
        if difficulties and self.difficulty not in difficulties:
            return False
        return True


class TaskLoader(ABC):
    """Abstract base class for loading tasks from various sources."""

    @abstractmethod
    def load(self, source: str | Path) -> list[Task]:
        """Load tasks from a source (file, directory, database, etc.)."""
        pass

    @abstractmethod
    def save(self, tasks: list[Task], destination: str | Path) -> None:
        """Save tasks to a destination."""
        pass


class JSONTaskLoader(TaskLoader):
    """Load tasks from JSON files.

    Supports:
    - Single file with {"tasks": [...]} or single task object
    - Directory of JSON files
    """

    def load(self, source: str | Path) -> list[Task]:
        """Load tasks from JSON file(s)."""
        path = Path(source)

        if path.is_file():
            with open(path) as f:
                data = json.load(f)

            # Handle both formats: {"tasks": [...]} or single task
            if isinstance(data, dict) and "tasks" in data:
                return [Task(**t) for t in data["tasks"]]
            elif isinstance(data, dict):
                return [Task(**data)]
            elif isinstance(data, list):
                return [Task(**t) for t in data]
            raise ValueError(f"Invalid JSON format in {path}")

        elif path.is_dir():
            tasks = []
            for file in sorted(path.glob("**/*.json")):
                tasks.extend(self.load(file))
            return tasks

        raise ValueError(f"Invalid source: {source}")

    def save(self, tasks: list[Task], destination: str | Path) -> None:
        """Save tasks to JSON file."""
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(
                {"tasks": [t.model_dump(mode="json") for t in tasks]},
                f,
                indent=2,
                default=str,
            )


class EvalSetMetadata(BaseModel):
    """Metadata for an evaluation set."""

    author: str | None = None
    version: str = "1.0.0"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    description: str | None = None
    tags: list[str] = Field(default_factory=list)


class EvalSet(BaseModel):
    """Collection of tasks for evaluation.

    An EvalSet groups related tasks together for batch evaluation.
    It also defines default configuration for all tasks in the set.

    Example:
        eval_set = EvalSet(
            name="Goal Decomposition Suite v1",
            tasks=tasks,
            default_num_runs=5,
            default_grader_ids=["quality", "personalization"],
        )
    """

    eval_set_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str

    # Tasks in this set
    tasks: list[Task] = Field(default_factory=list)

    # Default graders to use
    default_grader_ids: list[str] = Field(default_factory=list)

    # Metadata
    metadata: EvalSetMetadata = Field(default_factory=EvalSetMetadata)

    # Configuration
    default_num_runs: int = 1  # For pass@k
    default_timeout_seconds: float = 300.0

    def filter_tasks(
        self,
        tags: list[str] | None = None,
        categories: list[str] | None = None,
        difficulties: list[str] | None = None,
        max_tasks: int | None = None,
    ) -> list[Task]:
        """Filter tasks by criteria."""
        filtered = [
            t for t in self.tasks
            if t.matches_filter(tags=tags, categories=categories, difficulties=difficulties)
        ]
        if max_tasks:
            filtered = filtered[:max_tasks]
        return filtered

    def filtered_eval_set(
        self,
        tags: list[str] | None = None,
        categories: list[str] | None = None,
        difficulties: list[str] | None = None,
        max_tasks: int | None = None,
    ) -> "EvalSet":
        """Return a new EvalSet with only tasks matching the filter criteria."""
        filtered = self.filter_tasks(
            tags=tags, categories=categories,
            difficulties=difficulties, max_tasks=max_tasks,
        )
        return EvalSet(
            name=self.name,
            tasks=filtered,
            default_grader_ids=self.default_grader_ids,
            metadata=self.metadata,
            default_num_runs=self.default_num_runs,
            default_timeout_seconds=self.default_timeout_seconds,
        )

    def add_task(self, task: Task) -> None:
        """Add a task to the set."""
        self.tasks.append(task)
        self.metadata.updated_at = datetime.utcnow()

    def remove_task(self, task_id: str) -> bool:
        """Remove a task by ID. Returns True if found and removed."""
        original_len = len(self.tasks)
        self.tasks = [t for t in self.tasks if t.task_id != task_id]
        if len(self.tasks) < original_len:
            self.metadata.updated_at = datetime.utcnow()
            return True
        return False

    def get_task(self, task_id: str) -> Task | None:
        """Get a task by ID."""
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None
