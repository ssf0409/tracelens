"""HTTP API adapter for evaluating HTTP-based agents.

Provides an AgentAdapter that communicates with agents via HTTP endpoints,
with built-in auth, retry with exponential backoff, and response parsing.

Requires the `http` optional dependency: pip install eval-kit[http]
"""

import asyncio
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from eval_kit.core.task import Task
from eval_kit.core.transcript import StepType, ToolCall, Transcript, TranscriptStep
from eval_kit.execution.agent_adapter import AgentAdapter

try:
    import httpx
except ImportError as e:
    raise ImportError(
        "httpx is required for HTTPAPIAdapter. "
        "Install it with: pip install eval-kit[http]"
    ) from e


class AuthScheme(str, Enum):
    """Supported authentication schemes."""
    BEARER = "bearer"
    API_KEY = "api_key"
    CUSTOM = "custom"


class AuthConfig(BaseModel):
    """Authentication configuration for HTTP requests."""
    scheme: AuthScheme = AuthScheme.BEARER
    token: str | None = None
    api_key_header: str = "X-API-Key"
    api_key_value: str | None = None
    custom_headers: dict[str, str] = Field(default_factory=dict)

    def apply_to_headers(self, headers: dict[str, str]) -> dict[str, str]:
        """Apply auth config to a headers dict, returning the updated dict."""
        if self.scheme == AuthScheme.BEARER and self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        elif self.scheme == AuthScheme.API_KEY and self.api_key_value:
            headers[self.api_key_header] = self.api_key_value
        elif self.scheme == AuthScheme.CUSTOM:
            headers.update(self.custom_headers)
        return headers


class RetryConfig(BaseModel):
    """Retry configuration with exponential backoff."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    backoff_factor: float = 2.0
    retry_on_status_codes: list[int] = Field(default_factory=lambda: [429, 500, 502, 503, 504])


class HTTPAdapterConfig(BaseModel):
    """Full configuration for HTTPAPIAdapter."""
    base_url: str
    endpoint: str = "/"
    method: str = "POST"
    timeout: float = 30.0
    auth: AuthConfig | None = None
    retry: RetryConfig = Field(default_factory=RetryConfig)
    extra_headers: dict[str, str] = Field(default_factory=dict)


class HTTPAPIAdapter(AgentAdapter):
    """Adapter that invokes agents via HTTP API calls.

    Supports authentication, retry with exponential backoff, and
    customizable request/response handling.

    Example:
        config = HTTPAdapterConfig(
            base_url="https://api.example.com",
            endpoint="/v1/agent/run",
            auth=AuthConfig(scheme=AuthScheme.BEARER, token="sk-..."),
        )
        adapter = HTTPAPIAdapter(config)
        # Use with EvaluationRunner
    """

    def __init__(self, config: HTTPAdapterConfig) -> None:
        self.config = config
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """Lazy-create the httpx client on first use."""
        if self._client is None:
            headers: dict[str, str] = {"Content-Type": "application/json"}
            headers.update(self.config.extra_headers)
            if self.config.auth:
                self.config.auth.apply_to_headers(headers)

            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                headers=headers,
                timeout=self.config.timeout,
            )
        return self._client

    def build_request_body(self, task: Task) -> dict[str, Any]:
        """Build the HTTP request body from a task. Override to customize."""
        return dict(task.input_data)

    def parse_response_body(self, data: Any) -> Any:
        """Extract the agent's answer from response JSON. Override to customize."""
        return data

    async def _request_with_retry(
        self,
        task: Task,
        transcript: Transcript,
    ) -> Any:
        """Make HTTP request with exponential backoff retry."""
        client = self._get_client()
        retry_cfg = self.config.retry
        body = self.build_request_body(task)

        last_error: Exception | None = None

        for attempt in range(retry_cfg.max_retries + 1):
            try:
                response = await client.request(
                    method=self.config.method,
                    url=self.config.endpoint,
                    json=body,
                )

                # Record attempt as a tool call step
                transcript.add_step(TranscriptStep(
                    step_type=StepType.TOOL_CALL,
                    tool_call=ToolCall(
                        tool_name="http_request",
                        arguments={
                            "method": self.config.method,
                            "url": f"{self.config.base_url}{self.config.endpoint}",
                            "attempt": attempt + 1,
                        },
                        result={"status_code": response.status_code},
                    ),
                ))

                if response.status_code in retry_cfg.retry_on_status_codes:
                    last_error = httpx.HTTPStatusError(
                        f"HTTP {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                    if attempt < retry_cfg.max_retries:
                        delay = min(
                            retry_cfg.base_delay * (retry_cfg.backoff_factor ** attempt),
                            retry_cfg.max_delay,
                        )
                        await asyncio.sleep(delay)
                        continue
                    raise last_error

                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError:
                raise
            except Exception as exc:
                last_error = exc
                transcript.add_step(TranscriptStep(
                    step_type=StepType.ERROR,
                    error=f"HTTP request attempt {attempt + 1} failed: {exc}",
                ))
                if attempt < retry_cfg.max_retries:
                    delay = min(
                        retry_cfg.base_delay * (retry_cfg.backoff_factor ** attempt),
                        retry_cfg.max_delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise

        raise last_error or RuntimeError("All retry attempts exhausted")

    async def run(self, task: Task) -> Transcript:
        """Invoke the HTTP agent and return a transcript."""
        transcript = self.start_transcript(task)

        try:
            raw_response = await self._request_with_retry(task, transcript)
            result = self.parse_response_body(raw_response)

            transcript.final_output = result
            transcript.add_step(TranscriptStep(
                step_type=StepType.AGENT_OUTPUT,
                content=result,
            ))
        except Exception as exc:
            self.record_error(transcript, exc)
            raise
        finally:
            transcript.completed_at = datetime.utcnow()

        return transcript

    async def close(self) -> None:
        """Close the underlying httpx client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def teardown(self, task: Task, transcript: Transcript | None) -> None:
        """Close the HTTP client on teardown."""
        await self.close()
