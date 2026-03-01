"""Tests for evaluation runner module."""

import asyncio
from datetime import UTC, datetime

from eval_kit.core.grader import CodeGrader
from eval_kit.core.task import EvalSet, Task
from eval_kit.core.transcript import Transcript
from eval_kit.core.trial import TrialStatus
from eval_kit.execution.agent_adapter import AgentAdapter, SimpleAdapter
from eval_kit.execution.runner import EvaluationRunner, RunnerConfig


class _PassGrader(CodeGrader):
    """Grader that always passes with a fixed score."""

    def __init__(self, score: float = 0.8):
        super().__init__("pass_grader")
        self._score = score

    def compute_metrics(self, transcript: Transcript, task: Task) -> dict[str, float]:
        return {"quality": self._score}

    def determine_pass(self, metrics: dict[str, float], task: Task) -> tuple[bool, float]:
        return True, self._score


class _FailGrader(CodeGrader):
    """Grader that always fails."""

    def __init__(self):
        super().__init__("fail_grader")

    def compute_metrics(self, transcript: Transcript, task: Task) -> dict[str, float]:
        return {"quality": 0.1}

    def determine_pass(self, metrics: dict[str, float], task: Task) -> tuple[bool, float]:
        return False, 0.1


class _ExplodingGrader(CodeGrader):
    """Grader that raises an exception."""

    def __init__(self):
        super().__init__("exploding_grader")

    def compute_metrics(self, transcript: Transcript, task: Task) -> dict[str, float]:
        raise RuntimeError("grader error")

    def determine_pass(self, metrics: dict[str, float], task: Task) -> tuple[bool, float]:
        return False, 0.0


def _make_eval_set(n_tasks: int = 2) -> EvalSet:
    tasks = [
        Task(task_id=f"task-{i}", name=f"Task {i}", input_data={"n": i})
        for i in range(n_tasks)
    ]
    return EvalSet(name="Test Suite", tasks=tasks)


async def _echo_fn(input_data: dict) -> dict:
    return {"echo": input_data}


class TestRunnerConfig:
    def test_defaults(self):
        config = RunnerConfig()
        assert config.num_runs == 1
        assert config.max_concurrency == 5
        assert config.timeout_seconds == 300.0
        assert config.fail_fast is False


class TestEvaluationRunner:
    async def test_basic_run(self):
        """Runner produces one trial per task × run."""
        adapter = SimpleAdapter(_echo_fn)
        grader = _PassGrader()
        config = RunnerConfig(num_runs=3)

        runner = EvaluationRunner(adapter, [grader], config)
        batch = await runner.run(_make_eval_set(2))

        # 2 tasks × 3 runs = 6 trials
        assert batch.total_count == 6
        assert batch.all_complete
        assert batch.started_at is not None
        assert batch.completed_at is not None

    async def test_all_trials_graded(self):
        """Every completed trial gets an outcome from each grader."""
        adapter = SimpleAdapter(_echo_fn)
        graders = [_PassGrader(0.9), _FailGrader()]
        config = RunnerConfig(num_runs=1)

        runner = EvaluationRunner(adapter, graders, config)
        batch = await runner.run(_make_eval_set(1))

        trial = batch.trials[0]
        assert len(trial.outcomes) == 2
        assert trial.outcomes[0].grader_id == "pass_grader"
        assert trial.outcomes[1].grader_id == "fail_grader"

    async def test_trial_id_correctness(self):
        """Outcomes get the trial's ID via add_outcome(), not transcript's task_id."""
        adapter = SimpleAdapter(_echo_fn)
        runner = EvaluationRunner(adapter, [_PassGrader()])
        batch = await runner.run(_make_eval_set(1))

        trial = batch.trials[0]
        for outcome in trial.outcomes:
            assert outcome.trial_id == trial.trial_id

    async def test_timeout_handling(self):
        """Trials that exceed timeout are marked TIMEOUT."""
        async def slow_fn(input_data: dict) -> dict:
            await asyncio.sleep(10)
            return {}

        adapter = SimpleAdapter(slow_fn)
        config = RunnerConfig(timeout_seconds=0.05)
        runner = EvaluationRunner(adapter, [_PassGrader()], config)
        batch = await runner.run(_make_eval_set(1))

        trial = batch.trials[0]
        assert trial.status == TrialStatus.TIMEOUT
        assert trial.error_message is not None
        assert len(trial.outcomes) == 0  # No grading on timeout

    async def test_adapter_error_handling(self):
        """Adapter exceptions result in FAILED status with no grading."""
        async def broken_fn(input_data: dict) -> dict:
            raise RuntimeError("agent crashed")

        adapter = SimpleAdapter(broken_fn)
        runner = EvaluationRunner(adapter, [_PassGrader()])
        batch = await runner.run(_make_eval_set(1))

        trial = batch.trials[0]
        assert trial.status == TrialStatus.FAILED
        assert "agent crashed" in trial.error_message
        assert len(trial.outcomes) == 0

    async def test_grader_error_handling(self):
        """Grader exceptions produce failed outcomes (not trial failures)."""
        adapter = SimpleAdapter(_echo_fn)
        runner = EvaluationRunner(adapter, [_ExplodingGrader()])
        batch = await runner.run(_make_eval_set(1))

        trial = batch.trials[0]
        assert trial.status == TrialStatus.COMPLETED  # Trial itself succeeded
        assert len(trial.outcomes) == 1
        assert trial.outcomes[0].passed is False
        assert trial.outcomes[0].score == 0.0
        assert "grader error" in trial.outcomes[0].feedback

    async def test_concurrency_limit(self):
        """Max concurrency is respected."""
        max_concurrent = 0
        current_concurrent = 0

        async def tracking_fn(input_data: dict) -> dict:
            nonlocal max_concurrent, current_concurrent
            current_concurrent += 1
            max_concurrent = max(max_concurrent, current_concurrent)
            await asyncio.sleep(0.01)
            current_concurrent -= 1
            return {}

        adapter = SimpleAdapter(tracking_fn)
        config = RunnerConfig(num_runs=1, max_concurrency=2)
        runner = EvaluationRunner(adapter, [_PassGrader()], config)
        await runner.run(_make_eval_set(5))

        assert max_concurrent <= 2

    async def test_pass_rate(self):
        """TrialBatch pass_rate reflects actual grading results."""
        adapter = SimpleAdapter(_echo_fn)
        runner = EvaluationRunner(adapter, [_FailGrader()])
        batch = await runner.run(_make_eval_set(3))

        assert batch.pass_rate == 0.0

    async def test_pass_results_by_task(self):
        """get_pass_results_by_task groups correctly across runs."""
        adapter = SimpleAdapter(_echo_fn)
        config = RunnerConfig(num_runs=3)
        runner = EvaluationRunner(adapter, [_PassGrader()], config)
        batch = await runner.run(_make_eval_set(2))

        results = batch.get_pass_results_by_task()
        assert len(results) == 2
        for task_id, passes in results.items():
            assert len(passes) == 3
            assert all(passes)


# --- Lifecycle hooks integration with runner ---


class _LifecycleTracker(AgentAdapter):
    """Adapter that tracks lifecycle call order for runner integration tests."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.setup_error: Exception | None = None
        self.run_error: Exception | None = None
        self.teardown_error: Exception | None = None

    async def setup(self, task: Task) -> None:
        self.calls.append("setup")
        if self.setup_error:
            raise self.setup_error

    async def run(self, task: Task) -> Transcript:
        self.calls.append("run")
        if self.run_error:
            raise self.run_error
        transcript = self.start_transcript(task)
        transcript.final_output = "ok"
        transcript.completed_at = datetime.now(UTC)
        return transcript

    async def teardown(self, task: Task, transcript: Transcript | None) -> None:
        self.calls.append("teardown")
        if self.teardown_error:
            raise self.teardown_error


class TestRunnerLifecycleHooks:
    """Tests for lifecycle hooks integration with EvaluationRunner."""

    async def test_happy_path_order(self):
        """setup -> run -> teardown on success."""
        adapter = _LifecycleTracker()
        runner = EvaluationRunner(adapter, [_PassGrader()])
        batch = await runner.run(_make_eval_set(1))

        assert adapter.calls == ["setup", "run", "teardown"]
        assert batch.trials[0].status == TrialStatus.COMPLETED

    async def test_setup_failure_skips_run(self):
        """Setup failure skips run, teardown still called."""
        adapter = _LifecycleTracker()
        adapter.setup_error = RuntimeError("setup boom")

        runner = EvaluationRunner(adapter, [_PassGrader()])
        batch = await runner.run(_make_eval_set(1))

        assert adapter.calls == ["setup", "teardown"]
        assert batch.trials[0].status == TrialStatus.FAILED
        assert "Setup failed" in batch.trials[0].error_message

    async def test_run_failure_still_calls_teardown(self):
        """Run failure still calls teardown."""
        adapter = _LifecycleTracker()
        adapter.run_error = ValueError("run boom")

        runner = EvaluationRunner(adapter, [_PassGrader()])
        batch = await runner.run(_make_eval_set(1))

        assert adapter.calls == ["setup", "run", "teardown"]
        assert batch.trials[0].status == TrialStatus.FAILED
        assert "run boom" in batch.trials[0].error_message

    async def test_teardown_failure_on_success_marks_failed(self):
        """Teardown failure on an otherwise-successful trial marks it FAILED."""
        adapter = _LifecycleTracker()
        adapter.teardown_error = RuntimeError("teardown boom")

        runner = EvaluationRunner(adapter, [_PassGrader()])
        batch = await runner.run(_make_eval_set(1))

        assert adapter.calls == ["setup", "run", "teardown"]
        assert batch.trials[0].status == TrialStatus.FAILED
        assert "Teardown failed" in batch.trials[0].error_message

    async def test_both_run_and_teardown_fail(self):
        """Both run and teardown failures concatenate error messages."""
        adapter = _LifecycleTracker()
        adapter.run_error = ValueError("run boom")
        adapter.teardown_error = RuntimeError("teardown boom")

        runner = EvaluationRunner(adapter, [_PassGrader()])
        batch = await runner.run(_make_eval_set(1))

        trial = batch.trials[0]
        assert trial.status == TrialStatus.FAILED
        assert "run boom" in trial.error_message
        assert "Teardown also failed" in trial.error_message

    async def test_timeout_still_calls_teardown(self):
        """Timeout still calls teardown."""
        adapter = _LifecycleTracker()

        # Override run to sleep
        async def slow_run(task: Task) -> Transcript:
            adapter.calls.append("run")
            await asyncio.sleep(10)
            return adapter.start_transcript(task)

        adapter.run = slow_run  # type: ignore[assignment]

        config = RunnerConfig(timeout_seconds=0.05)
        runner = EvaluationRunner(adapter, [_PassGrader()], config)
        batch = await runner.run(_make_eval_set(1))

        assert "teardown" in adapter.calls
        assert batch.trials[0].status == TrialStatus.TIMEOUT
