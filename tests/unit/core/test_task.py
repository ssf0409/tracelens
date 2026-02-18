"""Tests for task module."""

import json
import pytest
from pathlib import Path
from datetime import datetime

from eval_kit.core.task import (
    Task,
    TaskExpectation,
    TaskLoader,
    JSONTaskLoader,
    EvalSet,
    EvalSetMetadata,
)


class TestTask:
    """Tests for Task model."""

    def test_task_creation_minimal(self):
        """Test creating a task with minimal fields."""
        task = Task(
            name="Test Task",
            input_data={"goal": "Test goal"},
        )

        assert task.name == "Test Task"
        assert task.input_data == {"goal": "Test goal"}
        assert task.task_id  # Auto-generated
        assert task.description is None
        assert task.expectation is None
        assert task.tags == []
        assert task.difficulty is None
        assert task.timeout_seconds == 300.0

    def test_task_creation_full(self, sample_task: Task):
        """Test creating a task with all fields."""
        assert sample_task.task_id == "test-task-001"
        assert sample_task.name == "Test Goal Decomposition"
        assert sample_task.category == "programming"
        assert "python" in sample_task.tags
        assert sample_task.difficulty == "medium"
        assert sample_task.expectation is not None
        assert sample_task.expectation.expected_metrics == {"quality": 0.8}

    def test_task_matches_filter_by_tags(self, sample_task: Task):
        """Test task filtering by tags."""
        assert sample_task.matches_filter(tags=["python"])
        assert sample_task.matches_filter(tags=["beginner"])
        assert sample_task.matches_filter(tags=["python", "java"])  # OR logic
        assert not sample_task.matches_filter(tags=["java"])

    def test_task_matches_filter_by_category(self, sample_task: Task):
        """Test task filtering by category."""
        assert sample_task.matches_filter(categories=["programming"])
        assert not sample_task.matches_filter(categories=["fitness"])

    def test_task_matches_filter_by_difficulty(self, sample_task: Task):
        """Test task filtering by difficulty."""
        assert sample_task.matches_filter(difficulties=["medium"])
        assert sample_task.matches_filter(difficulties=["easy", "medium"])
        assert not sample_task.matches_filter(difficulties=["hard"])

    def test_task_matches_filter_combined(self, sample_task: Task):
        """Test combined filtering."""
        assert sample_task.matches_filter(
            tags=["python"],
            categories=["programming"],
            difficulties=["medium"],
        )
        assert not sample_task.matches_filter(
            tags=["python"],
            categories=["fitness"],  # Wrong category
        )


class TestTaskExpectation:
    """Tests for TaskExpectation model."""

    def test_expectation_minimal(self):
        """Test minimal expectation."""
        exp = TaskExpectation()
        assert exp.expected_output is None
        assert exp.validation_hints == {}

    def test_expectation_with_metrics(self):
        """Test expectation with metrics."""
        exp = TaskExpectation(
            expected_metrics={"accuracy": 0.9, "speed": 0.5},
            metric_thresholds={"accuracy": 0.8},
        )
        assert exp.expected_metrics["accuracy"] == 0.9
        assert exp.metric_thresholds["accuracy"] == 0.8


class TestJSONTaskLoader:
    """Tests for JSONTaskLoader."""

    def test_load_single_task(self, tmp_path: Path):
        """Test loading a single task from JSON."""
        task_file = tmp_path / "task.json"
        task_file.write_text(json.dumps({
            "name": "Test Task",
            "input_data": {"goal": "Test"},
        }))

        loader = JSONTaskLoader()
        tasks = loader.load(task_file)

        assert len(tasks) == 1
        assert tasks[0].name == "Test Task"

    def test_load_tasks_array(self, tmp_path: Path):
        """Test loading tasks from array format."""
        task_file = tmp_path / "tasks.json"
        task_file.write_text(json.dumps({
            "tasks": [
                {"name": "Task 1", "input_data": {"goal": "Goal 1"}},
                {"name": "Task 2", "input_data": {"goal": "Goal 2"}},
            ]
        }))

        loader = JSONTaskLoader()
        tasks = loader.load(task_file)

        assert len(tasks) == 2
        assert tasks[0].name == "Task 1"
        assert tasks[1].name == "Task 2"

    def test_load_from_directory(self, tmp_path: Path):
        """Test loading tasks from directory."""
        (tmp_path / "task1.json").write_text(json.dumps({
            "name": "Task 1", "input_data": {"goal": "Goal 1"}
        }))
        (tmp_path / "task2.json").write_text(json.dumps({
            "name": "Task 2", "input_data": {"goal": "Goal 2"}
        }))

        loader = JSONTaskLoader()
        tasks = loader.load(tmp_path)

        assert len(tasks) == 2

    def test_save_tasks(self, tmp_path: Path):
        """Test saving tasks to JSON."""
        tasks = [
            Task(name="Task 1", input_data={"goal": "Goal 1"}),
            Task(name="Task 2", input_data={"goal": "Goal 2"}),
        ]

        loader = JSONTaskLoader()
        output_file = tmp_path / "output.json"
        loader.save(tasks, output_file)

        # Verify saved content
        loaded = loader.load(output_file)
        assert len(loaded) == 2
        assert loaded[0].name == "Task 1"


class TestEvalSet:
    """Tests for EvalSet model."""

    def test_eval_set_creation(self):
        """Test creating an eval set."""
        tasks = [
            Task(name="Task 1", input_data={"goal": "Goal 1"}, category="a"),
            Task(name="Task 2", input_data={"goal": "Goal 2"}, category="b"),
        ]

        eval_set = EvalSet(
            name="Test Suite",
            tasks=tasks,
            default_num_runs=5,
        )

        assert eval_set.name == "Test Suite"
        assert len(eval_set.tasks) == 2
        assert eval_set.default_num_runs == 5

    def test_eval_set_filter_tasks(self):
        """Test filtering tasks in eval set."""
        tasks = [
            Task(name="Task 1", input_data={}, category="a", tags=["x"]),
            Task(name="Task 2", input_data={}, category="b", tags=["y"]),
            Task(name="Task 3", input_data={}, category="a", tags=["x", "y"]),
        ]

        eval_set = EvalSet(name="Test", tasks=tasks)

        # Filter by category
        filtered = eval_set.filter_tasks(categories=["a"])
        assert len(filtered) == 2

        # Filter by tags
        filtered = eval_set.filter_tasks(tags=["y"])
        assert len(filtered) == 2

        # Combined filter
        filtered = eval_set.filter_tasks(categories=["a"], tags=["y"])
        assert len(filtered) == 1
        assert filtered[0].name == "Task 3"

    def test_eval_set_add_remove_task(self):
        """Test adding and removing tasks."""
        eval_set = EvalSet(name="Test", tasks=[])

        task = Task(task_id="t1", name="Task 1", input_data={})
        eval_set.add_task(task)
        assert len(eval_set.tasks) == 1

        removed = eval_set.remove_task("t1")
        assert removed is True
        assert len(eval_set.tasks) == 0

        removed = eval_set.remove_task("nonexistent")
        assert removed is False

    def test_eval_set_get_task(self):
        """Test getting a task by ID."""
        task = Task(task_id="t1", name="Task 1", input_data={})
        eval_set = EvalSet(name="Test", tasks=[task])

        found = eval_set.get_task("t1")
        assert found is not None
        assert found.name == "Task 1"

        not_found = eval_set.get_task("nonexistent")
        assert not_found is None

    def test_eval_set_filtered_eval_set(self):
        """Test filtered_eval_set returns a new EvalSet with filtered tasks."""
        tasks = [
            Task(name="Task 1", input_data={}, category="a", tags=["x"]),
            Task(name="Task 2", input_data={}, category="b", tags=["y"]),
            Task(name="Task 3", input_data={}, category="a", tags=["x", "y"]),
        ]

        eval_set = EvalSet(
            name="Test Suite",
            tasks=tasks,
            default_num_runs=5,
            default_grader_ids=["g1"],
        )

        result = eval_set.filtered_eval_set(categories=["a"])
        assert isinstance(result, EvalSet)
        assert len(result.tasks) == 2
        assert result.name == "Test Suite"
        assert result.default_num_runs == 5
        assert result.default_grader_ids == ["g1"]
