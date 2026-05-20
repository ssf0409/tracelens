"""Plugin loading via dotted import paths.

Provides a thin importlib wrapper for loading adapter and grader classes
at runtime from dotted path strings like "myproject.eval.graders.QualityGrader".
"""

import importlib
from typing import Any, cast


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
