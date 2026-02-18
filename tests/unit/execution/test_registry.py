"""Tests for registry module."""

import pytest

from eval_kit.execution.registry import load_class, instantiate


class TestLoadClass:
    def test_load_builtin_class(self):
        """Can load a class from stdlib."""
        cls = load_class("collections.OrderedDict")
        from collections import OrderedDict
        assert cls is OrderedDict

    def test_load_project_class(self):
        """Can load a class from eval_kit itself."""
        cls = load_class("eval_kit.core.task.Task")
        from eval_kit.core.task import Task
        assert cls is Task

    def test_invalid_path_no_module(self):
        """Raises ImportError for a path with no module separator."""
        with pytest.raises(ImportError, match="no module"):
            load_class("JustAClassName")

    def test_missing_module(self):
        """Raises ImportError for a nonexistent module."""
        with pytest.raises((ImportError, ModuleNotFoundError)):
            load_class("nonexistent.module.Class")

    def test_missing_class(self):
        """Raises AttributeError for a missing class in a valid module."""
        with pytest.raises(AttributeError):
            load_class("eval_kit.core.task.NonexistentClass")


class TestInstantiate:
    def test_instantiate_with_kwargs(self):
        """Can instantiate a class with keyword arguments."""
        obj = instantiate(
            "eval_kit.core.task.Task",
            name="Test",
            input_data={"a": 1},
        )
        from eval_kit.core.task import Task
        assert isinstance(obj, Task)
        assert obj.name == "Test"

    def test_instantiate_no_kwargs(self):
        """Can instantiate a class with no arguments."""
        obj = instantiate("collections.OrderedDict")
        from collections import OrderedDict
        assert isinstance(obj, OrderedDict)
