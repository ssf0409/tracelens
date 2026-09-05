"""Exercise the CI diagnostic entry point without Claude credentials or API calls."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / ".github/scripts/claude_diagnostics.py"


def run_diagnostics(tmp_path: Path, content: str | None) -> subprocess.CompletedProcess[str]:
    execution_file = tmp_path / "execution.json"
    if content is not None:
        execution_file.write_text(content)
    return subprocess.run(
        [sys.executable, str(SCRIPT)],
        env={
            **os.environ,
            "CLAUDE_EXECUTION_FILE": str(execution_file),
            "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"),
        },
        capture_output=True,
        text=True,
        check=True,
    )


@pytest.mark.parametrize(
    ("status", "hint"),
    [
        (400, "request rejected"),
        (401, "authentication rejected"),
        (403, "access denied"),
        (404, "model or endpoint"),
        (429, "rate or usage limit"),
        (500, "provider error"),
        (503, "provider error"),
    ],
)
def test_failure_reports_status_without_raw_content(tmp_path, status, hint):
    messages = [
        {"type": "system", "subtype": "init", "model": "PRIVATE_MODEL"},
        {
            "type": "result",
            "subtype": "success",
            "is_error": True,
            "result": f"API Error: {status} PRIVATE_TOKEN\n::error::INJECTED_COMMAND",
        },
    ]
    result = run_diagnostics(tmp_path, json.dumps(messages))
    assert "Result: failed" in result.stdout
    assert f"HTTP {status}" in result.stdout
    assert hint in result.stdout
    assert "PRIVATE" not in result.stdout
    assert "INJECTED_COMMAND" not in result.stdout
    assert result.stderr == ""
    assert (tmp_path / "summary.md").read_text() == result.stdout


def test_assistant_api_error_and_expired_oauth_hint(tmp_path):
    messages = [
        {
            "type": "assistant",
            "error": "authentication_failed",
            "message": {
                "api_error_status": 401,
                "content": [{"type": "text", "text": "OAuth token has expired. SECRET"}],
            },
        },
    ]
    result = run_diagnostics(tmp_path, json.dumps(messages))
    assert "Result: unavailable" in result.stdout
    assert "HTTP 401" in result.stdout
    assert "OAuth token expired" in result.stdout
    assert "SECRET" not in result.stdout


def test_error_result_array_and_malformed_records(tmp_path):
    messages = [
        None,
        1,
        [],
        {
            "type": "result",
            "subtype": "error_during_execution",
            "errors": [None, "API Error: 429 PRIVATE"],
        },
    ]
    result = run_diagnostics(tmp_path, json.dumps(messages))
    assert "Result: failed" in result.stdout
    assert "HTTP 429" in result.stdout
    assert "PRIVATE" not in result.stdout


@pytest.mark.parametrize("content", [None, "{PRIVATE_INVALID_JSON", "null", "{}", "42"])
def test_missing_or_invalid_file_does_not_expose_content(tmp_path, content):
    result = run_diagnostics(tmp_path, content)
    assert "Execution diagnostics unavailable" in result.stdout
    assert "PRIVATE" not in result.stdout
    assert "Traceback" not in result.stderr


def test_ignores_success_text_tool_output_and_non_error_assistant(tmp_path):
    messages = [
        {
            "type": "assistant",
            "message": {
                "api_error_status": 401,
                "content": [{"type": "text", "text": "API Error: 401 OAuth token has expired"}],
            },
        },
        {"type": "user", "content": [{"type": "tool_result", "content": "API Error: 403"}]},
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": "API Error: 429 SECRET",
        },
    ]
    result = run_diagnostics(tmp_path, json.dumps(messages))
    assert "Result: success" in result.stdout
    assert "HTTP" not in result.stdout
    assert "OAuth" not in result.stdout
    assert "SECRET" not in result.stdout


def test_unknown_failure_never_prints_arbitrary_fields(tmp_path):
    messages = [
        {
            "type": "result",
            "subtype": "SECRET",
            "is_error": True,
            "result": "SECRET",
            "errors": "SECRET",
        },
        {
            "type": "assistant",
            "error": "SECRET",
            "message": {
                "api_error_status": "SECRET",
                "content": [None, "SECRET", {"type": "text", "text": {"SECRET": "SECRET"}}],
            },
        },
    ]
    result = run_diagnostics(tmp_path, json.dumps(messages))
    assert "Result: failed" in result.stdout
    assert "No recognized API error" in result.stdout
    assert "SECRET" not in result.stdout
