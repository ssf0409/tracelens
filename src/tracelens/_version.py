"""Package version helpers."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("tracelens")
except PackageNotFoundError:  # pragma: no cover - source tree without install
    __version__ = "0.0.0+unknown"
