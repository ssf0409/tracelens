"""Tests for registry module."""

import pytest

from tracelens.execution.registry import instantiate, load_class


class TestLoadClass:
    def test_load_builtin_class(self):
        """Can load a class from stdlib."""
        cls = load_class("collections.OrderedDict")
        from collections import OrderedDict

        assert cls is OrderedDict

    def test_load_project_class(self):
        """Can load a class from tracelens itself."""
        cls = load_class("tracelens.core.task.Task")
        from tracelens.core.task import Task

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
            load_class("tracelens.core.task.NonexistentClass")


class TestInstantiate:
    def test_instantiate_with_kwargs(self):
        """Can instantiate a class with keyword arguments."""
        obj = instantiate(
            "tracelens.core.task.Task",
            name="Test",
            input_data={"a": 1},
        )
        from tracelens.core.task import Task

        assert isinstance(obj, Task)
        assert obj.name == "Test"

    def test_instantiate_no_kwargs(self):
        """Can instantiate a class with no arguments."""
        obj = instantiate("collections.OrderedDict")
        from collections import OrderedDict

        assert isinstance(obj, OrderedDict)


class TestPluginsAreLoadedFromTheSourceOnDisk:
    """The code that runs is the code on disk, whatever the bytecode cache says.

    Python treats a cached ``.pyc`` as valid while the source's size and its
    mtime in whole seconds are unchanged. An operator who edits a plugin to
    the same length and reruns within the second would otherwise have the
    previous plugin measured and reported as the current one. The condition
    is pinned here with ``os.utime`` so the test is deterministic, not timing
    dependent.
    """

    PLUGIN_V1 = 'class Thing:\n    VALUE = ">= 1:"\n'
    PLUGIN_V2 = 'class Thing:\n    VALUE = ">= 3:"\n'  # same length on purpose

    @staticmethod
    def _run(code: str, cwd) -> str:
        import os
        import subprocess
        import sys

        env = {**os.environ, "PYTHONPATH": str(cwd)}
        env.pop("PYTHONDONTWRITEBYTECODE", None)  # the cache is what is under test
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()

    def test_a_same_size_edit_within_the_same_second_is_still_loaded(self, tmp_path):
        import importlib.util
        import os
        from pathlib import Path

        plugin = tmp_path / "plug.py"
        plugin.write_text(self.PLUGIN_V1)
        via_registry = "from tracelens.execution.registry import load_class; print(load_class('plug.Thing').VALUE)"
        via_importlib = "import importlib; print(importlib.import_module('plug').Thing.VALUE)"

        assert self._run(via_registry, tmp_path) == ">= 1:"
        cached = Path(importlib.util.cache_from_source(str(plugin)))
        assert cached.is_file(), "the first load must have populated the cache"

        # The edit: same length, and pinned to the mtime second the cache recorded.
        before = plugin.stat().st_mtime
        plugin.write_text(self.PLUGIN_V2)
        os.utime(plugin, (before, before))

        # Control -- plain importlib is fooled. This is the defect, and it is
        # what makes the assertion below capable of failing.
        assert self._run(via_importlib, tmp_path) == ">= 1:"
        # The product loads what is on disk.
        assert self._run(via_registry, tmp_path) == ">= 3:"

    def test_the_cache_is_left_hash_checked_so_the_timestamp_cannot_lie_again(self, tmp_path):
        import importlib.util
        from pathlib import Path

        plugin = tmp_path / "plug.py"
        plugin.write_text(self.PLUGIN_V1)
        self._run(
            "from tracelens.execution.registry import load_class; load_class('plug.Thing')",
            tmp_path,
        )
        self._run(
            "from tracelens.execution.registry import load_class; load_class('plug.Thing')",
            tmp_path,
        )
        cached = Path(importlib.util.cache_from_source(str(plugin)))
        # PEP 552 header: bytes 4-8 are flags; bit 0 = hash-based, bit 1 = check the source.
        flags = int.from_bytes(cached.read_bytes()[4:8], "little")
        assert flags & 0b11 == 0b11

    def test_a_module_already_imported_is_never_swapped_underneath(self, tmp_path, monkeypatch):
        import sys

        from tracelens.execution.registry import _refresh_cached_bytecode

        plugin = tmp_path / "live_plug.py"
        plugin.write_text(self.PLUGIN_V1)
        monkeypatch.syspath_prepend(str(tmp_path))
        first = load_class("live_plug.Thing")
        plugin.write_text(self.PLUGIN_V2)
        _refresh_cached_bytecode("live_plug")  # must be a no-op for a live module
        assert load_class("live_plug.Thing") is first and first.VALUE == ">= 1:"
        sys.modules.pop("live_plug", None)

    def test_a_source_that_no_longer_compiles_raises_the_real_error(self, tmp_path):
        import subprocess
        import sys

        plugin = tmp_path / "plug.py"
        plugin.write_text(self.PLUGIN_V1)
        self._run(
            "from tracelens.execution.registry import load_class; load_class('plug.Thing')",
            tmp_path,
        )
        plugin.write_text("class Thing(\n")  # broken
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from tracelens.execution.registry import load_class; load_class('plug.Thing')",
            ],
            cwd=tmp_path,
            env={"PYTHONPATH": str(tmp_path), "PATH": ""},
            capture_output=True,
            text=True,
            timeout=120,
        )
        # Not the stale bytecode running happily: the syntax error from the source.
        assert result.returncode != 0 and "SyntaxError" in result.stderr
