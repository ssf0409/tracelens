"""Integration coverage for the dry-run LLM provider example."""

import subprocess
import sys


def test_llm_provider_examples_run_without_api_keys() -> None:
    result = subprocess.run(
        [sys.executable, "examples/llm_provider_examples.py"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "openai" in result.stdout
    assert "anthropic" in result.stdout
    assert "mode=dry-run" in result.stdout
    assert "score=0.90" in result.stdout
