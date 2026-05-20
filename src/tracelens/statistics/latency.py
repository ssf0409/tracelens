"""Latency analysis for streaming transcripts.

Computes first-token latency, throughput, and inter-token interval
statistics from streaming events captured in Transcripts.
"""

from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from tracelens.core.transcript import StreamingEventType, Transcript


class LatencyMetrics(BaseModel):
    """Latency metrics for a single transcript's streaming data."""

    first_token_ms: float | None = None
    time_to_complete_ms: float | None = None
    tokens_per_second: float | None = None
    inter_token_mean_ms: float | None = None
    inter_token_p50_ms: float | None = None
    inter_token_p95_ms: float | None = None
    inter_token_p99_ms: float | None = None
    total_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class AggregateLatencyMetrics(BaseModel):
    """Aggregated latency metrics across multiple transcripts."""

    count: int = 0
    mean_first_token_ms: float | None = None
    mean_time_to_complete_ms: float | None = None
    mean_tokens_per_second: float | None = None
    p50_first_token_ms: float | None = None
    p95_first_token_ms: float | None = None
    p99_first_token_ms: float | None = None
    per_transcript: list[LatencyMetrics] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class LatencyAnalyzer:
    """Analyzes streaming latency from transcript events."""

    def analyze(self, transcript: Transcript) -> LatencyMetrics:
        """Compute latency metrics for a single transcript."""
        if not transcript.has_streaming_data:
            return LatencyMetrics()

        token_events = [
            e for e in transcript.streaming_events
            if e.event_type == StreamingEventType.TOKEN
        ]

        if not token_events:
            return LatencyMetrics()

        first_token_ms = token_events[0].timestamp_ms
        last_event = transcript.streaming_events[-1]
        time_to_complete_ms = last_event.timestamp_ms

        # Each TOKEN event represents at least 1 token for throughput estimation,
        # even if token_count wasn't explicitly set by the adapter.
        total_tokens = sum(e.token_count or 1 for e in token_events)

        # Compute TPS using generation window (first token to last event),
        # excluding idle time before the first token arrived.
        generation_ms = time_to_complete_ms - first_token_ms
        generation_seconds = generation_ms / 1000.0
        tokens_per_second = (
            total_tokens / generation_seconds
            if generation_seconds > 0
            else None
        )

        # Inter-token intervals
        inter_token_mean_ms: float | None = None
        inter_token_p50_ms: float | None = None
        inter_token_p95_ms: float | None = None
        inter_token_p99_ms: float | None = None

        if len(token_events) >= 2:
            timestamps = [e.timestamp_ms for e in token_events]
            intervals = np.diff(timestamps)
            inter_token_mean_ms = float(np.mean(intervals))
            inter_token_p50_ms = float(np.percentile(intervals, 50))
            inter_token_p95_ms = float(np.percentile(intervals, 95))
            inter_token_p99_ms = float(np.percentile(intervals, 99))

        return LatencyMetrics(
            first_token_ms=first_token_ms,
            time_to_complete_ms=time_to_complete_ms,
            tokens_per_second=tokens_per_second,
            inter_token_mean_ms=inter_token_mean_ms,
            inter_token_p50_ms=inter_token_p50_ms,
            inter_token_p95_ms=inter_token_p95_ms,
            inter_token_p99_ms=inter_token_p99_ms,
            total_tokens=total_tokens,
        )

    def analyze_batch(self, transcripts: list[Transcript]) -> AggregateLatencyMetrics:
        """Compute aggregate latency metrics across multiple transcripts."""
        per_transcript = [self.analyze(t) for t in transcripts]
        streaming = [m for m in per_transcript if m.first_token_ms is not None]

        if not streaming:
            return AggregateLatencyMetrics(count=0, per_transcript=per_transcript)

        first_tokens = [m.first_token_ms for m in streaming if m.first_token_ms is not None]
        completions = [
            m.time_to_complete_ms for m in streaming
            if m.time_to_complete_ms is not None
        ]
        tps_values = [m.tokens_per_second for m in streaming if m.tokens_per_second is not None]

        ft_array = np.array(first_tokens)

        return AggregateLatencyMetrics(
            count=len(streaming),
            mean_first_token_ms=float(np.mean(ft_array)),
            mean_time_to_complete_ms=float(np.mean(completions)) if completions else None,
            mean_tokens_per_second=float(np.mean(tps_values)) if tps_values else None,
            p50_first_token_ms=float(np.percentile(ft_array, 50)),
            p95_first_token_ms=float(np.percentile(ft_array, 95)),
            p99_first_token_ms=float(np.percentile(ft_array, 99)),
            per_transcript=per_transcript,
        )
