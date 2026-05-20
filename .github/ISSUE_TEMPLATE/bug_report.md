---
name: Bug report
about: Report a defect with a minimal reproduction
title: "[bug] "
labels: bug
assignees: ''
---

## What happened

<!-- A clear, concise description of the bug. -->

## What you expected

<!-- What did you expect to happen instead? -->

## Minimal reproduction

<!--
Smallest possible code or eval-set snippet that reproduces the issue.
If it's CI-related, paste the relevant workflow + run URL.
-->

```python
# repro
```

## Environment

- tracelens version: `pip show tracelens | grep Version`
- Python version: `python --version`
- OS: (macOS 14 / Ubuntu 22.04 / Windows 11 / ...)
- Install method: `pip` / `uv` / git clone
- Optional extras installed: `[llm]` / `[http]` / `[dev]` / none

## Logs / traceback

<details>
<summary>Full traceback</summary>

```
<paste here>
```

</details>

## Anything else?

<!--
e.g., does it reproduce with `num_runs=1`? Does it reproduce on a fresh venv?
Does the same DecisionSpec fingerprint appear across runs?
-->
