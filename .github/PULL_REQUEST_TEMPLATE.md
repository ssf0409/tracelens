<!--
Thanks for contributing to tracelens!
Please keep one logical change per PR. Split branches that touch
multiple layers (e.g. adapter + statistics) into separate PRs.
-->

## Summary

<!-- One paragraph: what does this PR change, from the user's perspective? -->

## Motivation

<!--
Why is this change needed? What problem does it solve?
What alternatives did you consider and why did you reject them?
-->

## Type of change

- [ ] Bug fix
- [ ] New feature (non-breaking)
- [ ] Breaking change (please flag in CHANGELOG under [Unreleased] → Changed)
- [ ] Docs / examples / benchmarks
- [ ] Internal refactor (no user-visible change)

## Verification

- [ ] `pytest -q` passes locally
- [ ] `ruff check src/ tests/` passes
- [ ] `mypy src/tracelens/` passes (strict)
- [ ] New / changed code is covered by tests
- [ ] If touching `DecisionSpec`, baseline, or regression logic: backwards-compat note added to `CHANGELOG.md`
- [ ] If adding to `src/tracelens/__init__.py`: this symbol is intended as stable public API

## Linked issues

<!-- Closes #123 / Refs #456 -->

## Notes for the reviewer

<!-- Anything subtle, controversial, or worth flagging. -->
