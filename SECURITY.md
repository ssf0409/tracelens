# Security Policy

## Supported Versions

TraceLens is currently in `0.x` alpha. Only the latest minor release receives
security fixes; older `0.x` lines are not patched.

| Version | Supported |
| ------- | --------- |
| 0.1.x   | Yes       |
| < 0.1   | No        |

Once `1.0` ships, this table will be updated to a rolling N-1 minor policy.

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security reports.**

Email **ssf0409@gmail.com** with:

- A description of the vulnerability and its impact.
- A minimal reproduction (code snippet, malformed input, or attack scenario).
- The affected version (`pip show tracelens`) and Python version.
- Whether you'd like to be credited in the fix announcement.

You can expect:

- An acknowledgement within **5 business days**.
- A triage assessment (severity + estimated patch ETA) within **10 business
  days**.
- A coordinated-disclosure window of **up to 90 days** before the report is
  made public, extendable by mutual agreement for complex issues.

## Scope

In scope:

- Code execution, privilege escalation, or sandbox escape via untrusted task
  inputs (e.g. malicious `Task` JSON, malicious agent transcripts).
- Information disclosure via grader prompts (e.g. prompt-injection that
  leaks API keys or other transcripts).
- Denial-of-service in the runner, statistics, or baseline-management code
  paths reachable from a normal evaluation run.
- Tampering with `DecisionSpec` fingerprints in a way that breaks
  reproducibility guarantees.

Out of scope:

- Vulnerabilities in third-party LLM providers, sandboxes, or transports
  that tracelens merely calls into. Report those upstream.
- Denial-of-service achieved purely by configuring large `num_runs`,
  `concurrency`, or memory limits — those are user-controlled inputs.
- Issues in example code under `examples/` or `benchmarks/` that are
  illustrative rather than production surface area.

## Disclosure

After a fix is released, the advisory will be published as a GitHub
Security Advisory on the repository, and the fixed version + CVE (if
assigned) will be noted in `CHANGELOG.md`.
