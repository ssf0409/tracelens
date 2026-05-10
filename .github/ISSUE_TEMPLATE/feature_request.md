---
name: Feature request
about: Propose a new capability or API change
title: "[feat] "
labels: enhancement
assignees: ''
---

## The agent / eval scenario you're trying to support

<!--
Start with the use case, not the API.
What kind of agent? What grading dimension? What guarantee do you need?
-->

## What's missing today

<!-- Why can't you express this with the current eval-kit surface? -->

## Proposed API or behavior

<!--
Sketch the smallest API change you can imagine.
Code snippets > prose.
-->

```python
# proposed usage
```

## Alternatives you considered

<!-- e.g. subclassing existing graders, writing a custom adapter, doing it out-of-band. -->

## Compatibility considerations

- [ ] Does this require a new `DecisionSpec` field? (If yes, fingerprint impact?)
- [ ] Does this change baseline / regression semantics?
- [ ] Does this add a new optional dependency?
- [ ] Could this be a separate package that depends on eval-kit instead of living in core?
