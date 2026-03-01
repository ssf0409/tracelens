"""Tests for HTTP API adapter module."""

import pytest

import httpx
from unittest.mock import AsyncMock, patch

from eval_kit.core.task import Task
from eval_kit.core.transcript import StepType
from eval_kit.execution.http_adapter import (
    AuthConfig,
    AuthScheme,
    HTTPAdapterConfig,
    HTTPAPIAdapter,
    RetryConfig,
)


@pytest.fixture
def task() -> Task:
    return Task(task_id="t1", name="Test", input_data={"prompt": "hello"})


@pytest.fixture
def config() -> HTTPAdapterConfig:
    return HTTPAdapterConfig(
        base_url="https://api.example.com",
        endpoint="/v1/run",
        timeout=10.0,
        retry=RetryConfig(max_retries=2, base_delay=0.001, max_delay=0.01),
    )


def _mock_response(status_code: int = 200, json_data: dict | None = None) -> httpx.Response:
    """Create a mock httpx response."""
    json_data = json_data or {"result": "ok"}
    request = httpx.Request("POST", "https://api.example.com/v1/run")
    response = httpx.Response(
        status_code=status_code,
        json=json_data,
        request=request,
    )
    return response


class TestAuthConfig:
    def test_bearer_auth(self):
        auth = AuthConfig(scheme=AuthScheme.BEARER, token="sk-test")
        headers: dict[str, str] = {}
        auth.apply_to_headers(headers)
        assert headers["Authorization"] == "Bearer sk-test"

    def test_api_key_auth(self):
        auth = AuthConfig(
            scheme=AuthScheme.API_KEY,
            api_key_header="X-My-Key",
            api_key_value="key123",
        )
        headers: dict[str, str] = {}
        auth.apply_to_headers(headers)
        assert headers["X-My-Key"] == "key123"

    def test_custom_auth(self):
        auth = AuthConfig(
            scheme=AuthScheme.CUSTOM,
            custom_headers={"X-Custom": "val1", "X-Other": "val2"},
        )
        headers: dict[str, str] = {}
        auth.apply_to_headers(headers)
        assert headers == {"X-Custom": "val1", "X-Other": "val2"}


class TestHTTPAPIAdapter:
    async def test_successful_request(self, config: HTTPAdapterConfig, task: Task):
        adapter = HTTPAPIAdapter(config)
        mock_response = _mock_response(200, {"answer": "world"})

        with patch.object(
            httpx.AsyncClient, "request", new_callable=AsyncMock, return_value=mock_response
        ):
            transcript = await adapter.run(task)

        assert transcript.final_output == {"answer": "world"}
        assert transcript.task_id == "t1"
        assert not transcript.has_errors

        # Should have a TOOL_CALL step + AGENT_OUTPUT step
        tool_steps = transcript.get_steps_by_type(StepType.TOOL_CALL)
        assert len(tool_steps) == 1
        assert tool_steps[0].tool_call.tool_name == "http_request"

        output_steps = transcript.get_steps_by_type(StepType.AGENT_OUTPUT)
        assert len(output_steps) == 1

        await adapter.close()

    async def test_retry_on_429(self, config: HTTPAdapterConfig, task: Task):
        adapter = HTTPAPIAdapter(config)
        responses = [
            _mock_response(429),
            _mock_response(429),
            _mock_response(200, {"answer": "ok"}),
        ]
        call_count = 0

        async def mock_request(*args, **kwargs):
            nonlocal call_count
            idx = min(call_count, len(responses) - 1)
            call_count += 1
            return responses[idx]

        with patch.object(httpx.AsyncClient, "request", side_effect=mock_request):
            transcript = await adapter.run(task)

        assert transcript.final_output == {"answer": "ok"}
        # 3 attempts recorded as TOOL_CALL steps
        tool_steps = transcript.get_steps_by_type(StepType.TOOL_CALL)
        assert len(tool_steps) == 3

        await adapter.close()

    async def test_retry_exhausted_raises(self, config: HTTPAdapterConfig, task: Task):
        adapter = HTTPAPIAdapter(config)
        # All responses are 500 → retries exhaust
        mock_resp = _mock_response(500)

        with patch.object(
            httpx.AsyncClient, "request", new_callable=AsyncMock, return_value=mock_resp
        ):
            with pytest.raises(httpx.HTTPStatusError):
                await adapter.run(task)

        await adapter.close()

    async def test_custom_body_builder(self, config: HTTPAdapterConfig, task: Task):
        """Override build_request_body to customize the payload."""

        class CustomAdapter(HTTPAPIAdapter):
            def build_request_body(self, task: Task) -> dict:
                return {"custom_input": task.input_data["prompt"], "version": 2}

        adapter = CustomAdapter(config)
        mock_resp = _mock_response(200, {"result": "custom"})

        captured_kwargs: dict = {}

        async def capture_request(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return mock_resp

        with patch.object(httpx.AsyncClient, "request", side_effect=capture_request):
            transcript = await adapter.run(task)

        assert captured_kwargs["json"] == {"custom_input": "hello", "version": 2}
        assert transcript.final_output == {"result": "custom"}

        await adapter.close()

    async def test_custom_response_parser(self, config: HTTPAdapterConfig, task: Task):
        """Override parse_response_body to extract from nested response."""

        class NestedParser(HTTPAPIAdapter):
            def parse_response_body(self, data):
                return data["data"]["answer"]

        adapter = NestedParser(config)
        mock_resp = _mock_response(200, {"data": {"answer": "nested_result"}})

        with patch.object(
            httpx.AsyncClient, "request", new_callable=AsyncMock, return_value=mock_resp
        ):
            transcript = await adapter.run(task)

        assert transcript.final_output == "nested_result"

        await adapter.close()

    async def test_auth_headers_applied(self, task: Task):
        """Auth config is applied to the client headers."""
        config = HTTPAdapterConfig(
            base_url="https://api.example.com",
            endpoint="/v1/run",
            auth=AuthConfig(scheme=AuthScheme.BEARER, token="test-token"),
        )
        adapter = HTTPAPIAdapter(config)
        client = adapter._get_client()

        assert "Authorization" in client.headers
        assert client.headers["Authorization"] == "Bearer test-token"

        await adapter.close()

    async def test_teardown_closes_client(self, config: HTTPAdapterConfig, task: Task):
        """teardown() closes the underlying httpx client."""
        adapter = HTTPAPIAdapter(config)
        _ = adapter._get_client()  # Force client creation
        assert adapter._client is not None

        await adapter.teardown(task, None)
        assert adapter._client is None

    async def test_default_request_body(self, config: HTTPAdapterConfig, task: Task):
        """Default build_request_body returns task.input_data."""
        adapter = HTTPAPIAdapter(config)
        body = adapter.build_request_body(task)
        assert body == {"prompt": "hello"}
