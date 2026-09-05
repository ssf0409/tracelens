"""Print allowlisted failure diagnostics, never raw Claude messages or tool output."""

import json
import os
import re
from pathlib import Path
from typing import Any

HTTP_HINTS = {
    400: "request rejected; check provider/model configuration (not proof of an expired token)",
    401: "authentication rejected; check CLAUDE_CODE_OAUTH_TOKEN",
    403: "access denied; check account/model permissions",
    404: "model or endpoint unavailable; check model configuration",
    429: "rate or usage limit; check subscription capacity before retrying",
}


def failure_summary(messages: list[Any]) -> str:
    result_state = "unavailable"
    error_text: list[str] = []
    statuses: set[int] = set()
    for record in messages:
        if not isinstance(record, dict):
            continue
        if record.get("type") == "result":
            failed = record.get("is_error") is True or record.get("subtype") != "success"
            result_state = "failed" if failed else "success"
            if failed:
                error_text.append(record.get("result"))
                if isinstance(record.get("errors"), list):
                    error_text.extend(record["errors"])
        elif record.get("type") == "assistant" and record.get("error"):
            message = record.get("message")
            if not isinstance(message, dict):
                continue
            status = message.get("api_error_status")
            if type(status) is int and 400 <= status <= 599:
                statuses.add(status)
            content = message.get("content")
            if isinstance(content, list):
                error_text.extend(
                    block.get("text")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )

    # Inspect only error messages. Output is built entirely from fixed strings and
    # validated HTTP codes, so credentials and workflow commands cannot be echoed.
    text = "\n".join(part for part in error_text if isinstance(part, str)).lower()
    statuses.update(int(code) for code in re.findall(r"api error:\s*([45]\d{2})\b", text))
    lines = ["### Claude execution diagnostics", f"Result: {result_state}"]
    for status in sorted(statuses):
        hint = HTTP_HINTS.get(status, "provider error" if status >= 500 else "request failed")
        lines.append(f"HTTP {status}: {hint}.")
    if "oauth token has expired" in text or "oauth token is expired" in text:
        lines.append(
            "OAuth token expired: regenerate with claude setup-token and update the secret."
        )
    if "oauth access token is invalid" in text or "oauth token is invalid" in text:
        lines.append(
            "OAuth token invalid: regenerate with claude setup-token and update the secret."
        )
    if not statuses and result_state != "success":
        lines.append(
            "No recognized API error; inspect action setup logs. Raw output remains hidden."
        )
    lines.append("Execution status is not a code-review verdict; verify the review on the PR head.")
    return "\n".join(lines) + "\n"


def main() -> None:
    try:
        messages = json.loads(Path(os.environ.get("CLAUDE_EXECUTION_FILE", "")).read_text())
        if not isinstance(messages, list):
            raise ValueError("Expected an execution message array")
        summary = failure_summary(messages)
    except (OSError, ValueError):
        summary = "Execution diagnostics unavailable; inspect action setup logs. Raw output remains hidden.\n"
    print(summary, end="")
    if summary_path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(summary_path).open("a") as file:
            file.write(summary)


if __name__ == "__main__":
    main()
