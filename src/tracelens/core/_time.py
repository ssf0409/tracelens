"""Internal time utilities."""

from datetime import UTC, datetime


def utc_now() -> datetime:
    """Return current UTC time as a timezone-aware datetime.

    Replacement for the deprecated ``datetime.utcnow()``, which returned
    a naive datetime and is removed in Python 3.14.
    """
    return datetime.now(UTC)
