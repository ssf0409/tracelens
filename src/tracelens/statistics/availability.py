"""Explicit metric availability.

A statistic that could not be measured is *unavailable*, never zero. A
suite with one run per task cannot support pass@5 or pass^3; reporting
``1.0`` (a fallback) or ``0.0`` (no eligible task) for them would be read
as evidence. :class:`MetricValue` carries a metric's value together with
the evidence behind it (eligible and total task counts, the runs the
metric needs) so report renderers can show ``N/A`` with a reason.

See ``docs/statistical-contract.md`` ("Availability").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MetricValue:
    """A suite-level metric with its availability evidence.

    Attributes:
        name: Metric name, for example ``"pass@5"`` or ``"pass^3"``.
        value: The measured value, or ``None`` when unavailable.
        eligible_tasks: Tasks that contributed to the value, or ``None``
            when not recorded (legacy reports).
        total_tasks: Tasks in the input, eligible or not, or ``None`` when
            not recorded.
        required_runs: Gradable runs per task the metric needs (``k``).
        max_runs: The largest number of gradable runs any task has.
        reason: Why the metric is unavailable; ``None`` when available.
    """

    name: str
    value: float | None
    eligible_tasks: int | None = None
    total_tasks: int | None = None
    required_runs: int | None = None
    max_runs: int | None = None
    reason: str | None = None

    @property
    def available(self) -> bool:
        """Whether the metric was measured."""
        return self.value is not None

    def describe(self) -> str:
        """Render the value with its evidence, or ``N/A`` with the reason."""
        if self.value is None:
            parts = [f"N/A: {self.reason}" if self.reason else "N/A"]
            if self.total_tasks is not None:
                parts.append(
                    f"{self.eligible_tasks or 0}/{self.total_tasks} tasks eligible"
                )
            if self.max_runs is not None:
                parts.append(f"max {self.max_runs} gradable run(s) recorded")
            return "; ".join(parts)
        text = f"{self.value:.4f}"
        if self.total_tasks is not None:
            text += f" ({self.eligible_tasks}/{self.total_tasks} tasks)"
        return text

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe representation; ``available`` is included explicitly."""
        return {
            "name": self.name,
            "value": self.value,
            "available": self.available,
            "eligible_tasks": self.eligible_tasks,
            "total_tasks": self.total_tasks,
            "required_runs": self.required_runs,
            "max_runs": self.max_runs,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetricValue:
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            name=str(data["name"]),
            value=data.get("value"),
            eligible_tasks=data.get("eligible_tasks"),
            total_tasks=data.get("total_tasks"),
            required_runs=data.get("required_runs"),
            max_runs=data.get("max_runs"),
            reason=data.get("reason"),
        )

    @classmethod
    def legacy(cls, name: str, value: float | None) -> MetricValue:
        """Entry for a report written before availability was recorded.

        The value is shown as recorded; nothing is known about eligibility,
        so a zero may be an unavailable metric rather than a measured zero.
        """
        return cls(
            name=name,
            value=value,
            reason=None if value is not None else "not recorded",
        )


def unavailable_reason(kind: str, k: int, total_tasks: int) -> str:
    """Standard reason text for a k-based metric no task can support."""
    if total_tasks == 0:
        return "no tasks with gradable trials"
    return f"needs at least {k} {kind} per task"
