"""Tests for streaming latency analysis."""

import pytest

from eval_kit.core.transcript import (
    StreamingEvent,
    StreamingEventType,
    Transcript,
)
from eval_kit.statistics.latency import LatencyAnalyzer


def _make_streaming_transcript(
    token_timestamps: list[float],
    token_counts: list[int] | None = None,
) -> Transcript:
    """Build a transcript with streaming events at given timestamps."""
    t = Transcript(task_id="t1")

    t.add_streaming_event(StreamingEvent(
        event_type=StreamingEventType.STREAM_START,
        timestamp_ms=0.0,
    ))

    for i, ts in enumerate(token_timestamps):
        count = token_counts[i] if token_counts else 1
        t.add_streaming_event(StreamingEvent(
            event_type=StreamingEventType.TOKEN,
            timestamp_ms=ts,
            content=f"tok{i}",
            token_count=count,
        ))

    end_ts = token_timestamps[-1] + 10.0 if token_timestamps else 10.0
    t.add_streaming_event(StreamingEvent(
        event_type=StreamingEventType.STREAM_END,
        timestamp_ms=end_ts,
    ))

    return t


class TestTranscriptStreamingProperties:
    """Tests for streaming properties on Transcript."""

    def test_no_streaming_data(self):
        t = Transcript(task_id="t1")
        assert t.has_streaming_data is False
        assert t.first_token_latency_ms is None
        assert t.streaming_duration_ms is None
        assert t.streaming_token_count == 0

    def test_has_streaming_data(self):
        t = _make_streaming_transcript([50.0, 100.0])
        assert t.has_streaming_data is True

    def test_first_token_latency(self):
        t = _make_streaming_transcript([42.0, 80.0, 120.0])
        assert t.first_token_latency_ms == 42.0

    def test_streaming_duration(self):
        t = _make_streaming_transcript([50.0, 100.0])
        # Duration = last event (110.0) - first event (0.0) = 110.0
        assert t.streaming_duration_ms == 110.0

    def test_streaming_token_count(self):
        t = _make_streaming_transcript([50.0, 100.0, 150.0], token_counts=[3, 5, 2])
        assert t.streaming_token_count == 10

    def test_add_streaming_event(self):
        t = Transcript(task_id="t1")
        event = StreamingEvent(
            event_type=StreamingEventType.TOKEN,
            timestamp_ms=50.0,
            content="hello",
        )
        t.add_streaming_event(event)
        assert len(t.streaming_events) == 1
        assert t.streaming_events[0].content == "hello"


class TestLatencyAnalyzer:
    """Tests for LatencyAnalyzer."""

    def test_empty_transcript(self):
        analyzer = LatencyAnalyzer()
        result = analyzer.analyze(Transcript(task_id="t1"))
        assert result.first_token_ms is None
        assert result.total_tokens == 0

    def test_single_token(self):
        analyzer = LatencyAnalyzer()
        t = _make_streaming_transcript([50.0])
        result = analyzer.analyze(t)

        assert result.first_token_ms == 50.0
        assert result.total_tokens == 1
        # No inter-token intervals with just one token
        assert result.inter_token_mean_ms is None

    def test_multiple_tokens(self):
        analyzer = LatencyAnalyzer()
        t = _make_streaming_transcript([50.0, 100.0, 150.0, 200.0])
        result = analyzer.analyze(t)

        assert result.first_token_ms == 50.0
        assert result.total_tokens == 4
        # Inter-token intervals: [50, 50, 50], mean = 50
        assert result.inter_token_mean_ms == pytest.approx(50.0)
        assert result.inter_token_p50_ms == pytest.approx(50.0)

    def test_tokens_per_second(self):
        analyzer = LatencyAnalyzer()
        # 4 tokens over 210ms (last event = 200+10) → ~19.05 tok/s
        t = _make_streaming_transcript([50.0, 100.0, 150.0, 200.0])
        result = analyzer.analyze(t)

        assert result.tokens_per_second is not None
        assert result.tokens_per_second > 0

    def test_custom_token_counts(self):
        analyzer = LatencyAnalyzer()
        t = _make_streaming_transcript([50.0, 100.0], token_counts=[3, 5])
        result = analyzer.analyze(t)
        assert result.total_tokens == 8

    def test_batch_analysis(self):
        analyzer = LatencyAnalyzer()
        t1 = _make_streaming_transcript([50.0, 100.0])
        t2 = _make_streaming_transcript([30.0, 60.0, 90.0])
        t3 = Transcript(task_id="no-stream")  # Non-streaming

        agg = analyzer.analyze_batch([t1, t2, t3])

        assert agg.count == 2  # Only streaming transcripts
        assert agg.mean_first_token_ms is not None
        assert len(agg.per_transcript) == 3  # All transcripts in per_transcript

    def test_batch_empty(self):
        analyzer = LatencyAnalyzer()
        agg = analyzer.analyze_batch([])
        assert agg.count == 0

    def test_to_dict(self):
        analyzer = LatencyAnalyzer()
        t = _make_streaming_transcript([50.0, 100.0])
        result = analyzer.analyze(t)
        d = result.to_dict()

        assert "first_token_ms" in d
        assert "tokens_per_second" in d
        assert "total_tokens" in d
