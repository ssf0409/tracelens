"""Tests for HTTP API adapter module."""

import subprocess
import sys
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from pydantic import ValidationError

from tracelens.core.task import Task
from tracelens.core.transcript import StepType
from tracelens.execution.http_adapter import (
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

    def test_bearer_without_token_raises(self):
        with pytest.raises(ValidationError, match="requires a non-empty 'token'"):
            AuthConfig(scheme=AuthScheme.BEARER)

    def test_api_key_without_value_raises(self):
        with pytest.raises(ValidationError, match="requires a non-empty 'api_key_value'"):
            AuthConfig(scheme=AuthScheme.API_KEY)


class TestHTTPAPIAdapter:
    def test_package_import_without_http_extra(self):
        """Core package imports should not require the optional http extra."""
        script = """
import importlib.abc
import sys


class BlockHttpx(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "httpx" or fullname.startswith("httpx."):
            raise ModuleNotFoundError("No module named 'httpx'")
        return None


sys.meta_path.insert(0, BlockHttpx())
from tracelens import HTTPAPIAdapter, HTTPAdapterConfig

try:
    HTTPAPIAdapter(HTTPAdapterConfig(base_url="https://api.example.com"))
except ImportError as exc:
    assert "pip install tracelens[http]" in str(exc)
else:
    raise AssertionError("HTTPAPIAdapter should require httpx at instantiation")
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr

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

    async def test_close_closes_client(self, config: HTTPAdapterConfig, task: Task):
        """close() shuts down the underlying httpx client."""
        adapter = HTTPAPIAdapter(config)
        _ = adapter._get_client()  # Force client creation
        assert adapter._client is not None

        await adapter.close()
        assert adapter._client is None

    async def test_teardown_preserves_client(self, config: HTTPAdapterConfig, task: Task):
        """teardown() is a no-op to preserve connection pooling across trials."""
        adapter = HTTPAPIAdapter(config)
        _ = adapter._get_client()
        assert adapter._client is not None

        await adapter.teardown(task, None)
        assert adapter._client is not None

    async def test_default_request_body(self, config: HTTPAdapterConfig, task: Task):
        """Default build_request_body returns task.input_data."""
        adapter = HTTPAPIAdapter(config)
        body = adapter.build_request_body(task)
        assert body == {"prompt": "hello"}

    async def test_non_retryable_error_raises_immediately(
        self, config: HTTPAdapterConfig, task: Task
    ):
        """Non-network errors (e.g. ValueError) are not retried."""
        adapter = HTTPAPIAdapter(config)
        call_count = 0

        async def raise_value_error(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise ValueError("bad request body")

        with patch.object(httpx.AsyncClient, "request", side_effect=raise_value_error):
            with pytest.raises(ValueError, match="bad request body"):
                await adapter.run(task)

        # Should NOT have retried — only 1 call
        assert call_count == 1
        await adapter.close()

    async def test_connection_error_retried(
        self, config: HTTPAdapterConfig, task: Task
    ):
        """Network errors like ConnectError are retried."""
        adapter = HTTPAPIAdapter(config)
        call_count = 0

        async def flaky_connection(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise httpx.ConnectError("connection refused")
            return _mock_response(200, {"result": "recovered"})

        with patch.object(httpx.AsyncClient, "request", side_effect=flaky_connection):
            transcript = await adapter.run(task)

        assert call_count == 3
        assert transcript.final_output == {"result": "recovered"}
        await adapter.close()
