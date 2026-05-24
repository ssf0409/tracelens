"""Integration coverage for the dry-run LLM provider example."""

import os
import subprocess
import sys


def run_example(*args: str, **env: str) -> subprocess.CompletedProcess[str]:
    example_env = os.environ.copy()
    example_env.pop("TRACELENS_LIVE", None)
    example_env.pop("TRACELENS_PROVIDER", None)
    example_env.pop("OPENAI_API_KEY", None)
    example_env.pop("ANTHROPIC_API_KEY", None)
    example_env.update(env)
    return subprocess.run(
        [sys.executable, "examples/llm_provider_examples.py", *args],
        check=True,
        capture_output=True,
        env=example_env,
        text=True,
    )


def test_llm_provider_examples_run_all_providers_without_api_keys() -> None:
    result = run_example()

    assert "openai" in result.stdout
    assert "anthropic" in result.stdout
    assert "mode=dry-run" in result.stdout
    assert "score=0.90" in result.stdout


def test_llm_provider_examples_accept_provider_cli_arg() -> None:
    result = run_example("--provider", "openai")

    assert "openai" in result.stdout
    assert "anthropic" not in result.stdout
    assert "mode=dry-run" in result.stdout


def test_llm_provider_examples_use_provider_env_fallback() -> None:
    result = run_example(TRACELENS_PROVIDER="anthropic")

    assert "anthropic" in result.stdout
    assert "openai" not in result.stdout
    assert "mode=dry-run" in result.stdout


def test_llm_provider_examples_live_mode_requires_provider_key() -> None:
    env = os.environ.copy()
    env.pop("TRACELENS_LIVE", None)
    env.pop("OPENAI_API_KEY", None)

    result = subprocess.run(
        [
            sys.executable,
            "examples/llm_provider_examples.py",
            "--provider",
            "openai",
            "--live",
        ],
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode != 0
    assert "OPENAI_API_KEY is required for --provider openai --live" in result.stderr


def test_llm_provider_examples_live_env_fallback_requires_provider_key() -> None:
    env = os.environ.copy()
    env["TRACELENS_LIVE"] = "1"
    env.pop("OPENAI_API_KEY", None)

    result = subprocess.run(
        [sys.executable, "examples/llm_provider_examples.py", "--provider", "openai"],
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode != 0
    assert "OPENAI_API_KEY is required for --provider openai --live" in result.stderr
