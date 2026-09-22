"""Plugin loading via dotted import paths.

Provides a thin importlib wrapper for loading adapter and grader classes
at runtime from dotted path strings like "myproject.eval.graders.QualityGrader".

The one thing it adds to ``importlib`` is a guarantee: the code that runs is
the code on disk. Python decides a cached ``.pyc`` is still valid when the
source file's size and its modification time *in whole seconds* are both
unchanged -- a timestamp heuristic, not a content check. TraceLens loads
plugins that operators edit between runs, so an edit that keeps the file's
length and lands within a second of the previous import is invisible to that
check: the run executes the previous adapter's bytecode and reports its
outcomes as the current adapter's. For a tool whose job is to tell whether an
agent changed, measuring the wrong agent is the one failure it must not have.
"""

import contextlib
import importlib
import importlib.util
import os
import py_compile
import sys
from typing import Any, cast


def _refresh_cached_bytecode(module_path: str) -> None:
    """Make the next import of ``module_path`` execute the source on disk.

    Every module along the dotted path that has a ``.py`` source and a cached
    ``.pyc`` gets that cache rewritten as a *checked-hash* pyc (PEP 552),
    which Python validates against the source's content on every import
    rather than against its timestamp. Modules already imported by this
    process are left alone: swapping code underneath a live module is its own
    hazard, and a fresh ``tracelens`` process is what a rerun is.

    If the cache cannot be rewritten (a source that no longer compiles, a
    read-only tree) it is removed instead, so the import falls back to the
    source and surfaces the real error rather than running stale bytecode.
    """
    parts = module_path.split(".")
    for depth in range(1, len(parts) + 1):
        name = ".".join(parts[:depth])
        if name in sys.modules:
            continue
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, AttributeError, ValueError):
            return  # let import_module raise the real error
        if spec is None or not spec.origin or not spec.origin.endswith(".py"):
            continue
        cached = spec.cached
        if not cached or not os.path.isfile(cached):
            continue  # nothing stale to inherit
        try:
            py_compile.compile(
                spec.origin,
                cfile=cached,
                doraise=True,
                invalidation_mode=py_compile.PycInvalidationMode.CHECKED_HASH,
            )
        except (py_compile.PyCompileError, OSError):
            with contextlib.suppress(OSError):
                os.remove(cached)


def load_class(dotted_path: str) -> type:
    """Load a class from a dotted import path.

    Args:
        dotted_path: e.g. "myproject.eval.graders.QualityGrader"

    Returns:
        The class object

    Raises:
        ImportError: If the module cannot be imported
        AttributeError: If the class is not found in the module
    """
    module_path, _, class_name = dotted_path.rpartition(".")
    if not module_path:
        raise ImportError(f"Invalid dotted path (no module): {dotted_path}")
    _refresh_cached_bytecode(module_path)
    module = importlib.import_module(module_path)
    return cast(type, getattr(module, class_name))


def instantiate(dotted_path: str, **kwargs: Any) -> Any:
    """Load a class and instantiate it with the given kwargs.

    Args:
        dotted_path: e.g. "myproject.eval.graders.QualityGrader"
        **kwargs: Arguments to pass to the constructor

    Returns:
        An instance of the class
    """
    cls = load_class(dotted_path)
    return cls(**kwargs)
