"""Tests for evaluation runner module."""

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from tracelens.core.grader import CodeGrader
from tracelens.core.task import EvalSet, Task
from tracelens.core.transcript import Transcript
from tracelens.core.trial import InfraError, TrialStatus
from tracelens.execution.agent_adapter import AgentAdapter, SimpleAdapter
from tracelens.execution.runner import EvaluationRunner, RunnerConfig


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


# --- Infra-error vs task-failure classification (Track 2 / Anthropic) ---


class TestInfraErrorClassification:
    """Runner must distinguish infrastructure failures from task failures
    so the infra-error rate can be reported separately and regressions
    aren't spuriously attributed to the agent."""

    async def test_explicit_infra_error_marks_infra_error_status(self):
        """Adapters can raise InfraError to self-report an infra failure."""
        async def raises_infra(input_data: dict) -> dict:
            raise InfraError("upstream API unreachable")

        runner = EvaluationRunner(SimpleAdapter(raises_infra), [_PassGrader()])
        batch = await runner.run(_make_eval_set(1))

        trial = batch.trials[0]
        assert trial.status == TrialStatus.INFRA_ERROR
        assert trial.is_infra_failure is True
        assert "upstream API unreachable" in trial.error_message
        assert len(trial.outcomes) == 0  # No grading — we don't have a transcript
        # Batch-level aggregation surfaces it.
        assert batch.infra_error_count == 1
        assert batch.infra_error_rate == 1.0

    async def test_memory_error_classified_as_infra(self):
        """OOM kills are the canonical Anthropic case — always infra."""
        async def ooms(input_data: dict) -> dict:
            raise MemoryError("out of memory")

        runner = EvaluationRunner(SimpleAdapter(ooms), [_PassGrader()])
        batch = await runner.run(_make_eval_set(1))

        assert batch.trials[0].status == TrialStatus.INFRA_ERROR
        assert batch.infra_error_rate == 1.0

    async def test_connection_error_classified_as_infra(self):
        """Network failures are infra, not task-level."""
        async def network_down(input_data: dict) -> dict:
            raise ConnectionError("connection refused")

        runner = EvaluationRunner(SimpleAdapter(network_down), [_PassGrader()])
        batch = await runner.run(_make_eval_set(1))

        assert batch.trials[0].status == TrialStatus.INFRA_ERROR

    async def test_generic_runtime_error_stays_task_failure(self):
        """We don't want arbitrary RuntimeError bugs in the agent to
        silently inflate the infra-error rate and mask regressions."""
        async def generic_bug(input_data: dict) -> dict:
            raise RuntimeError("agent bug")

        runner = EvaluationRunner(SimpleAdapter(generic_bug), [_PassGrader()])
        batch = await runner.run(_make_eval_set(1))

        assert batch.trials[0].status == TrialStatus.FAILED
        assert batch.infra_error_count == 0
        # But still surfaced in the error message for debugging.
        assert "agent bug" in batch.trials[0].error_message

    async def test_setup_infra_error_classified_correctly(self):
        """Infra errors during setup should also be classified as INFRA_ERROR,
        not as a task failure — the agent never got the chance to run."""

        class _InfraFailSetup(AgentAdapter):
            async def setup(self, task: Task) -> None:
                raise InfraError("sandbox provisioning failed")

            async def run(self, task: Task) -> Transcript:
                return Transcript(task_id=task.task_id)

            async def teardown(
                self, task: Task, transcript: Transcript | None
            ) -> None:
                return

        runner = EvaluationRunner(_InfraFailSetup(), [_PassGrader()])
        batch = await runner.run(_make_eval_set(1))

        trial = batch.trials[0]
        assert trial.status == TrialStatus.INFRA_ERROR
        assert "sandbox provisioning failed" in trial.error_message

    async def test_mixed_batch_reports_partial_infra_rate(self):
        """A mix of healthy and infra-failed trials gives a clean
        infra_error_rate like 0.5, not just 0/1."""

        call_count = {"n": 0}

        async def flaky(input_data: dict) -> dict:
            call_count["n"] += 1
            if call_count["n"] % 2 == 0:
                raise InfraError("transient OOM")
            return {"ok": True}

        runner = EvaluationRunner(
            SimpleAdapter(flaky),
            [_PassGrader()],
            RunnerConfig(num_runs=4),
        )
        batch = await runner.run(_make_eval_set(1))

        assert batch.total_count == 4
        # Exactly the even-indexed calls raised, so half the batch is infra.
        assert batch.infra_error_count == 2
        assert batch.infra_error_rate == 0.5
        # The successful ones still pass-through to grading.
        assert batch.passed_count == 2


# --- Fail-fast semantics ---


class TestFailFast:
    """RunnerConfig.fail_fast stops scheduling new trials after the first
    execution failure (FAILED or INFRA_ERROR); queued trials are recorded
    as SKIPPED rather than silently dropped."""

    async def test_fail_fast_skips_pending_trials_after_failure(self):
        """After the first FAILED trial, queued trials are not executed;
        they are recorded as SKIPPED so the batch stays accountable."""
        calls: list[int] = []

        async def fail_first(input_data: dict) -> dict:
            calls.append(input_data["n"])
            if input_data["n"] == 0:
                raise RuntimeError("boom")
            return {"ok": True}

        config = RunnerConfig(fail_fast=True, max_concurrency=1)
        runner = EvaluationRunner(SimpleAdapter(fail_first), [_PassGrader()], config)
        batch = await runner.run(_make_eval_set(4))

        # Only the failing trial actually ran.
        assert calls == [0]
        # Every work item is still accounted for in the batch.
        assert batch.total_count == 4
        statuses = [t.status for t in batch.trials]
        assert statuses.count(TrialStatus.FAILED) == 1
        assert statuses.count(TrialStatus.SKIPPED) == 3
        assert batch.all_complete

        skipped = [t for t in batch.trials if t.status == TrialStatus.SKIPPED]
        for trial in skipped:
            assert "fail_fast" in trial.error_message
            assert trial.outcomes == []  # never ran, never graded
            assert trial.transcript is None

    async def test_fail_fast_triggers_on_infra_error(self):
        """INFRA_ERROR trials trigger fail-fast just like FAILED ones."""
        calls: list[int] = []

        async def infra_fail_first(input_data: dict) -> dict:
            calls.append(input_data["n"])
            if input_data["n"] == 0:
                raise InfraError("sandbox died")
            return {"ok": True}

        config = RunnerConfig(fail_fast=True, max_concurrency=1)
        runner = EvaluationRunner(
            SimpleAdapter(infra_fail_first), [_PassGrader()], config
        )
        batch = await runner.run(_make_eval_set(3))

        assert calls == [0]
        statuses = [t.status for t in batch.trials]
        assert statuses.count(TrialStatus.INFRA_ERROR) == 1
        assert statuses.count(TrialStatus.SKIPPED) == 2

    async def test_fail_fast_disabled_runs_everything(self):
        """Default (fail_fast=False): failures never stop the batch."""
        calls: list[int] = []

        async def always_fail(input_data: dict) -> dict:
            calls.append(input_data["n"])
            raise RuntimeError("boom")

        config = RunnerConfig(max_concurrency=1)
        runner = EvaluationRunner(SimpleAdapter(always_fail), [_PassGrader()], config)
        batch = await runner.run(_make_eval_set(3))

        assert len(calls) == 3
        assert all(t.status == TrialStatus.FAILED for t in batch.trials)

    async def test_fail_fast_ignores_graded_failures(self):
        """A trial that executes fine but fails grading is COMPLETED, not
        FAILED — it must not trip fail-fast, which targets execution errors."""
        config = RunnerConfig(fail_fast=True, max_concurrency=1)
        runner = EvaluationRunner(SimpleAdapter(_echo_fn), [_FailGrader()], config)
        batch = await runner.run(_make_eval_set(3))

        assert all(t.status == TrialStatus.COMPLETED for t in batch.trials)
        assert batch.pass_rate == 0.0

    async def test_resume_reruns_skipped_trials(self, tmp_path: Path):
        """SKIPPED trials in a checkpoint are not real results: resuming a
        fail-fast run must execute them instead of loading them as done."""
        ckpt = str(tmp_path / "checkpoint.json")

        async def fail_first(input_data: dict) -> dict:
            if input_data["n"] == 0:
                raise RuntimeError("boom")
            return {"ok": True}

        config = RunnerConfig(
            fail_fast=True, max_concurrency=1, checkpoint_path=ckpt
        )
        runner = EvaluationRunner(SimpleAdapter(fail_first), [_PassGrader()], config)
        first = await runner.run(_make_eval_set(3))
        assert sum(t.status == TrialStatus.SKIPPED for t in first.trials) == 2

        async def now_ok(input_data: dict) -> dict:
            return {"ok": True}

        config2 = RunnerConfig(
            fail_fast=True, max_concurrency=1, checkpoint_path=ckpt
        )
        runner2 = EvaluationRunner(SimpleAdapter(now_ok), [_PassGrader()], config2)
        second = await runner2.run(_make_eval_set(3))

        statuses = [t.status for t in second.trials]
        # The FAILED trial is a real result and stays loaded from checkpoint;
        # the two previously-skipped trials were re-run and completed.
        assert statuses.count(TrialStatus.FAILED) == 1
        assert statuses.count(TrialStatus.COMPLETED) == 2
        assert statuses.count(TrialStatus.SKIPPED) == 0
