"""Tests for infra-error retry policy in the runner.

Trials that end INFRA_ERROR carry no signal about the agent, so the
runner can re-attempt them (opt-in via ``RunnerConfig.max_infra_retries``).
Task-level failures must NOT retry — retrying FAILED trials would
silently launder agent flakiness out of the pass rate.
"""

import asyncio

from tracelens.core.task import EvalSet, Task
from tracelens.core.transcript import Transcript
from tracelens.core.trial import InfraError, TrialStatus
from tracelens.execution.agent_adapter import AgentAdapter, SimpleAdapter
from tracelens.execution.runner import EvaluationRunner, RunnerConfig


def _eval_set(n_tasks: int = 1) -> EvalSet:
    tasks = [
        Task(task_id=f"task-{i}", name=f"Task {i}", input_data={"n": i})
        for i in range(n_tasks)
    ]
    return EvalSet(name="s", tasks=tasks)


class _FlakyInfraAdapter(AgentAdapter):
    """Raises InfraError for the first ``fail_times`` run() calls per task."""

    def __init__(self, fail_times: int) -> None:
        self.fail_times = fail_times
        self.calls: dict[str, int] = {}

    async def run(self, task: Task) -> Transcript:
        n = self.calls.get(task.task_id, 0) + 1
        self.calls[task.task_id] = n
        if n <= self.fail_times:
            raise InfraError(f"transient failure #{n}")
        return Transcript(task_id=task.task_id, final_output={"ok": True})


class TestInfraRetry:
    async def test_default_config_does_not_retry(self):
        """max_infra_retries defaults to 0: existing behavior unchanged."""
        adapter = _FlakyInfraAdapter(fail_times=1)
        runner = EvaluationRunner(adapter, [])
        batch = await runner.run(_eval_set(1))

        assert adapter.calls == {"task-0": 1}
        trial = batch.trials[0]
        assert trial.status == TrialStatus.INFRA_ERROR
        assert trial.attempts == 1

    async def test_retries_until_success(self):
        """A transient infra error is retried and the trial completes."""
        adapter = _FlakyInfraAdapter(fail_times=2)
        config = RunnerConfig(max_infra_retries=2, infra_retry_backoff_seconds=0.0)
        runner = EvaluationRunner(adapter, [], config)
        batch = await runner.run(_eval_set(1))

        assert adapter.calls == {"task-0": 3}
        trial = batch.trials[0]
        assert trial.status == TrialStatus.COMPLETED
        assert trial.attempts == 3
        # The batch is clean: the retried-away errors don't count as infra.
        assert batch.infra_error_count == 0
        assert batch.total_count == 1

    async def test_exhausted_retries_keep_infra_error(self):
        """When every attempt hits infra, the trial stays INFRA_ERROR."""
        adapter = _FlakyInfraAdapter(fail_times=100)
        config = RunnerConfig(max_infra_retries=2, infra_retry_backoff_seconds=0.0)
        runner = EvaluationRunner(adapter, [], config)
        batch = await runner.run(_eval_set(1))

        assert adapter.calls == {"task-0": 3}
        trial = batch.trials[0]
        assert trial.status == TrialStatus.INFRA_ERROR
        assert trial.attempts == 3
        assert batch.infra_error_count == 1

    async def test_earlier_attempt_errors_recorded_in_metadata(self):
        """Errors from retried-away attempts stay inspectable on the trial."""
        adapter = _FlakyInfraAdapter(fail_times=2)
        config = RunnerConfig(max_infra_retries=2, infra_retry_backoff_seconds=0.0)
        runner = EvaluationRunner(adapter, [], config)
        batch = await runner.run(_eval_set(1))

        retried = batch.trials[0].metadata["infra_retry_errors"]
        assert len(retried) == 2
        assert "transient failure #1" in retried[0]
        assert "transient failure #2" in retried[1]

    async def test_task_failures_do_not_retry(self):
        """FAILED is an agent outcome — never retried, even when opted in."""
        call_count = {"n": 0}

        async def agent_bug(input_data: dict) -> dict:
            call_count["n"] += 1
            raise RuntimeError("agent bug")

        config = RunnerConfig(max_infra_retries=3, infra_retry_backoff_seconds=0.0)
        runner = EvaluationRunner(SimpleAdapter(agent_bug), [], config)
        batch = await runner.run(_eval_set(1))

        assert call_count["n"] == 1
        trial = batch.trials[0]
        assert trial.status == TrialStatus.FAILED
        assert trial.attempts == 1

    async def test_timeouts_do_not_retry(self):
        """TIMEOUT is a legitimate observation about the agent — no retry."""
        call_count = {"n": 0}

        async def too_slow(input_data: dict) -> dict:
            call_count["n"] += 1
            await asyncio.sleep(10)
            return {}

        config = RunnerConfig(
            timeout_seconds=0.05,
            max_infra_retries=3,
            infra_retry_backoff_seconds=0.0,
        )
        runner = EvaluationRunner(SimpleAdapter(too_slow), [], config)
        batch = await runner.run(_eval_set(1))

        assert call_count["n"] == 1
        assert batch.trials[0].status == TrialStatus.TIMEOUT

    async def test_setup_infra_error_also_retries(self):
        """Infra failures during setup are retried like run() failures."""

        class _FlakySetupAdapter(AgentAdapter):
            def __init__(self) -> None:
                self.setup_calls = 0

            async def setup(self, task: Task) -> None:
                self.setup_calls += 1
                if self.setup_calls == 1:
                    raise InfraError("sandbox provisioning failed")

            async def run(self, task: Task) -> Transcript:
                return Transcript(task_id=task.task_id, final_output={"ok": True})

        adapter = _FlakySetupAdapter()
        config = RunnerConfig(max_infra_retries=1, infra_retry_backoff_seconds=0.0)
        runner = EvaluationRunner(adapter, [], config)
        batch = await runner.run(_eval_set(1))

        assert adapter.setup_calls == 2
        trial = batch.trials[0]
        assert trial.status == TrialStatus.COMPLETED
        assert trial.attempts == 2

    async def test_retry_backoff_grows_exponentially(self, monkeypatch):
        """Backoff between attempts doubles, mirroring GraderConfig retry."""
        sleeps: list[float] = []
        real_sleep = asyncio.sleep

        async def recording_sleep(delay: float) -> None:
            sleeps.append(delay)
            await real_sleep(0)

        monkeypatch.setattr(
            "tracelens.execution.runner.asyncio.sleep", recording_sleep
        )

        adapter = _FlakyInfraAdapter(fail_times=100)
        config = RunnerConfig(max_infra_retries=2, infra_retry_backoff_seconds=0.1)
        runner = EvaluationRunner(adapter, [], config)
        await runner.run(_eval_set(1))

        assert sleeps == [0.1, 0.2]
