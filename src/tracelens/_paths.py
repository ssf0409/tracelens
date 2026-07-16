"""Shared filesystem path handling for Task loaders."""

from pathlib import Path


def source_files(source: str | Path, suffix: str) -> list[Path]:
    """Resolve a file or recursively sorted directory into source files."""
    path = Path(source)
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.glob(f"**/*{suffix}"))
    raise ValueError(f"Source is not a file or directory: {source!r}")


def prepare_destination_path(destination: str | Path) -> Path:
    """Normalize a destination and ensure its parent directory exists."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
