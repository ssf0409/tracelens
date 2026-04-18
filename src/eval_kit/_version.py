"""Single source of truth for the package version.

Kept in its own module so submodules (like reporting.generator) can
import the version without hitting a circular import through
``eval_kit/__init__.py``.
"""

__version__ = "0.1.0"
