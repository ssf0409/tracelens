"""Round-trip tests for CSVTaskLoader, JSONLTaskLoader, and HFDatasetLoader."""

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tracelens.loaders import CSVTaskLoader, HFDatasetLoader, JSONLTaskLoader
from tracelens.core.task import Task


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write a list of dicts to a JSONL file."""
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    """Write a list of dicts to a CSV file."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ===========================================================================
# JSONLTaskLoader
# ===========================================================================


class TestJSONLTaskLoader:
    """Tests for JSONLTaskLoader."""

    # -----------------------------------------------------------------------
    # load()
    # -----------------------------------------------------------------------

    def test_load_basic(self, tmp_path: Path) -> None:
        """Each line becomes one Task; default 'input' field is used."""
        jsonl_file = tmp_path / "tasks.jsonl"
        _write_jsonl(
            jsonl_file,
            [
                {"input": {"goal": "Write a haiku"}, "name": "Haiku task"},
                {"input": {"goal": "Summarise the article"}, "name": "Summary task"},
            ],
        )

        tasks = JSONLTaskLoader().load(jsonl_file)

        assert len(tasks) == 2
        assert tasks[0].name == "Haiku task"
        assert tasks[0].input_data == {"goal": "Write a haiku"}
        assert tasks[1].name == "Summary task"
        assert tasks[1].input_data == {"goal": "Summarise the article"}

    def test_load_custom_input_field(self, tmp_path: Path) -> None:
        """Custom *input_field* is mapped to Task.input_data."""
        jsonl_file = tmp_path / "custom.jsonl"
        _write_jsonl(
            jsonl_file,
            [{"prompt": "Translate to French", "name": "Translation"}],
        )

        tasks = JSONLTaskLoader(input_field="prompt").load(jsonl_file)

        assert len(tasks) == 1
        assert tasks[0].input_data == {"value": "Translate to French"}

    def test_load_metadata_fields_selected(self, tmp_path: Path) -> None:
        """Only requested *metadata_fields* end up in Task.metadata."""
        jsonl_file = tmp_path / "meta.jsonl"
        _write_jsonl(
            jsonl_file,
            [
                {
                    "input": {"goal": "Plan a trip"},
                    "name": "Travel planner",
                    "category": "travel",
                    "difficulty": "easy",
                    "secret": "should-be-excluded",
                },
            ],
        )

        tasks = JSONLTaskLoader(metadata_fields=["category"]).load(jsonl_file)

        assert len(tasks) == 1
        assert "category" in tasks[0].metadata
        assert "secret" not in tasks[0].metadata

    def test_load_metadata_all_by_default(self, tmp_path: Path) -> None:
        """When *metadata_fields* is None, non-reserved/non-input keys go into metadata."""
        jsonl_file = tmp_path / "all_meta.jsonl"
        _write_jsonl(
            jsonl_file,
            [{"input": {"goal": "Cook pasta"}, "name": "Cooking", "source": "kitchen"}],
        )

        tasks = JSONLTaskLoader().load(jsonl_file)

        assert tasks[0].metadata.get("source") == "kitchen"

    def test_load_scalar_input_wrapped(self, tmp_path: Path) -> None:
        """A scalar input value is wrapped as {'value': ...} for Task.input_data."""
        jsonl_file = tmp_path / "scalar.jsonl"
        _write_jsonl(jsonl_file, [{"input": "Just a string", "name": "Scalar task"}])

        tasks = JSONLTaskLoader().load(jsonl_file)

        assert tasks[0].input_data == {"value": "Just a string"}

    def test_load_blank_lines_skipped(self, tmp_path: Path) -> None:
        """Blank lines in the JSONL file are silently ignored."""
        jsonl_file = tmp_path / "blanks.jsonl"
        jsonl_file.write_text(
            '\n{"input": {"goal": "g1"}, "name": "T1"}\n\n{"input": {"goal": "g2"}, "name": "T2"}\n',
            encoding="utf-8",
        )

        tasks = JSONLTaskLoader().load(jsonl_file)

        assert len(tasks) == 2

    def test_load_from_directory(self, tmp_path: Path) -> None:
        """Loading from a directory collects all *.jsonl files recursively."""
        (tmp_path / "a.jsonl").write_text(
            json.dumps({"input": {"goal": "A"}, "name": "Task A"}) + "\n",
            encoding="utf-8",
        )
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "b.jsonl").write_text(
            json.dumps({"input": {"goal": "B"}, "name": "Task B"}) + "\n",
            encoding="utf-8",
        )

        tasks = JSONLTaskLoader().load(tmp_path)

        names = {t.name for t in tasks}
        assert names == {"Task A", "Task B"}

    def test_load_invalid_json_raises(self, tmp_path: Path) -> None:
        """A malformed line raises ValueError with a useful location hint."""
        bad_file = tmp_path / "bad.jsonl"
        bad_file.write_text("not valid json\n", encoding="utf-8")

        with pytest.raises(ValueError, match="invalid JSON"):
            JSONLTaskLoader().load(bad_file)

    def test_load_non_object_line_raises(self, tmp_path: Path) -> None:
        """A JSON array on a line (not an object) raises ValueError."""
        bad_file = tmp_path / "array.jsonl"
        bad_file.write_text("[1, 2, 3]\n", encoding="utf-8")

        with pytest.raises(ValueError, match="JSON object"):
            JSONLTaskLoader().load(bad_file)

    def test_load_missing_source_raises(self, tmp_path: Path) -> None:
        """A path that doesn't exist raises ValueError."""
        with pytest.raises(ValueError, match="not a file or directory"):
            JSONLTaskLoader().load(tmp_path / "ghost.jsonl")

    # -----------------------------------------------------------------------
    # save() — and round-trip
    # -----------------------------------------------------------------------

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        """save() creates intermediate directories automatically."""
        dest = tmp_path / "nested" / "dir" / "out.jsonl"
        tasks = [Task(name="T", input_data={"goal": "g"})]

        JSONLTaskLoader().save(tasks, dest)

        assert dest.exists()

    def test_roundtrip_basic(self, tmp_path: Path) -> None:
        """save() → load() preserves name and input_data."""
        original = [
            Task(name="Task One", input_data={"goal": "Write tests"}, category="dev"),
            Task(name="Task Two", input_data={"goal": "Review PR"}, category="dev"),
        ]
        dest = tmp_path / "output.jsonl"

        loader = JSONLTaskLoader()
        loader.save(original, dest)
        loaded = loader.load(dest)

        assert len(loaded) == len(original)
        for orig, reloaded in zip(original, loaded):
            assert reloaded.name == orig.name
            assert reloaded.input_data == orig.input_data

    def test_roundtrip_with_metadata(self, tmp_path: Path) -> None:
        """Metadata survives a save/load cycle."""
        original = [Task(name="T", input_data={"q": "x"}, metadata={"source": "web"})]
        dest = tmp_path / "meta_rt.jsonl"

        loader = JSONLTaskLoader()
        loader.save(original, dest)
        loaded = loader.load(dest)

        assert loaded[0].metadata.get("source") == "web"

    def test_roundtrip_custom_input_field(self, tmp_path: Path) -> None:
        """A custom input_field is written and re-read consistently."""
        original = [Task(name="T", input_data={"goal": "custom"})]
        dest = tmp_path / "custom_field.jsonl"
        loader = JSONLTaskLoader(input_field="prompt")

        loader.save(original, dest)

        # The raw file should use "prompt", not "input_data".
        raw = json.loads(dest.read_text(encoding="utf-8").strip())
        assert "prompt" in raw
        assert "input_data" not in raw

        # And loading with the same loader recovers the task correctly.
        loaded = loader.load(dest)
        assert loaded[0].input_data == {"goal": "custom"}

    def test_roundtrip_task_id_preserved(self, tmp_path: Path) -> None:
        """task_id is preserved across a round-trip."""
        original = [Task(task_id="fixed-id-42", name="T", input_data={})]
        dest = tmp_path / "ids.jsonl"

        loader = JSONLTaskLoader()
        loader.save(original, dest)
        loaded = loader.load(dest)

        assert loaded[0].task_id == "fixed-id-42"


# ===========================================================================
# CSVTaskLoader
# ===========================================================================


class TestCSVTaskLoader:
    """Tests for CSVTaskLoader."""

    # -----------------------------------------------------------------------
    # load()
    # -----------------------------------------------------------------------

    def test_load_basic(self, tmp_path: Path) -> None:
        """Each CSV row becomes one Task using the default 'input' column."""
        csv_file = tmp_path / "tasks.csv"
        _write_csv(
            csv_file,
            [
                {"input": "Explain photosynthesis", "name": "Biology 101"},
                {"input": "Describe the water cycle", "name": "Earth Science"},
            ],
            fieldnames=["input", "name"],
        )

        tasks = CSVTaskLoader().load(csv_file)

        assert len(tasks) == 2
        assert tasks[0].name == "Biology 101"
        assert tasks[0].input_data == {"value": "Explain photosynthesis"}
        assert tasks[1].name == "Earth Science"

    def test_load_dict_input_parsed(self, tmp_path: Path) -> None:
        """If the input cell is a JSON object string, it is parsed to a dict."""
        csv_file = tmp_path / "json_input.csv"
        _write_csv(
            csv_file,
            [{"input": json.dumps({"goal": "Build an API"}), "name": "API task"}],
            fieldnames=["input", "name"],
        )

        tasks = CSVTaskLoader().load(csv_file)

        assert tasks[0].input_data == {"goal": "Build an API"}

    def test_load_custom_input_col(self, tmp_path: Path) -> None:
        """Custom *input_col* maps that column to Task.input_data."""
        csv_file = tmp_path / "custom_col.csv"
        _write_csv(
            csv_file,
            [{"prompt": "Summarise this doc", "name": "Summary"}],
            fieldnames=["prompt", "name"],
        )

        tasks = CSVTaskLoader(input_col="prompt").load(csv_file)

        assert tasks[0].input_data == {"value": "Summarise this doc"}

    def test_load_metadata_cols_selected(self, tmp_path: Path) -> None:
        """Only *metadata_cols* columns land in Task.metadata."""
        csv_file = tmp_path / "selective_meta.csv"
        _write_csv(
            csv_file,
            [
                {
                    "input": "Classify sentiment",
                    "name": "Sentiment",
                    "lang": "en",
                    "source": "twitter",
                    "internal_id": "x99",
                },
            ],
            fieldnames=["input", "name", "lang", "source", "internal_id"],
        )

        tasks = CSVTaskLoader(metadata_cols=["lang", "source"]).load(csv_file)

        assert tasks[0].metadata == {"lang": "en", "source": "twitter"}
        assert "internal_id" not in tasks[0].metadata

    def test_load_metadata_all_by_default(self, tmp_path: Path) -> None:
        """When *metadata_cols* is None, non-reserved columns go to metadata."""
        csv_file = tmp_path / "default_meta.csv"
        _write_csv(
            csv_file,
            [{"input": "Do something", "name": "T", "extra": "yes"}],
            fieldnames=["input", "name", "extra"],
        )

        tasks = CSVTaskLoader().load(csv_file)

        assert tasks[0].metadata.get("extra") == "yes"

    def test_load_from_directory(self, tmp_path: Path) -> None:
        """Loading from a directory picks up all *.csv files recursively."""
        _write_csv(
            tmp_path / "a.csv",
            [{"input": "Task A", "name": "A"}],
            fieldnames=["input", "name"],
        )
        sub = tmp_path / "sub"
        sub.mkdir()
        _write_csv(
            sub / "b.csv",
            [{"input": "Task B", "name": "B"}],
            fieldnames=["input", "name"],
        )

        tasks = CSVTaskLoader().load(tmp_path)

        names = {t.name for t in tasks}
        assert names == {"A", "B"}

    def test_load_reserved_fields_forwarded(self, tmp_path: Path) -> None:
        """Reserved columns such as 'difficulty' are forwarded to Task directly."""
        csv_file = tmp_path / "reserved.csv"
        _write_csv(
            csv_file,
            [{"input": "Hard task", "name": "Tricky", "difficulty": "hard"}],
            fieldnames=["input", "name", "difficulty"],
        )

        tasks = CSVTaskLoader().load(csv_file)

        assert tasks[0].difficulty == "hard"

    def test_load_missing_source_raises(self, tmp_path: Path) -> None:
        """A non-existent path raises ValueError."""
        with pytest.raises(ValueError, match="not a file or directory"):
            CSVTaskLoader().load(tmp_path / "ghost.csv")

    # -----------------------------------------------------------------------
    # save() — and round-trip
    # -----------------------------------------------------------------------

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        """save() creates intermediate directories automatically."""
        dest = tmp_path / "nested" / "out.csv"
        CSVTaskLoader().save([Task(name="T", input_data={"goal": "x"})], dest)

        assert dest.exists()

    def test_save_empty_list_writes_empty_file(self, tmp_path: Path) -> None:
        """Saving an empty list produces an empty file without error."""
        dest = tmp_path / "empty.csv"
        CSVTaskLoader().save([], dest)

        assert dest.exists()
        assert dest.read_text(encoding="utf-8") == ""

    def test_roundtrip_basic(self, tmp_path: Path) -> None:
        """save() → load() preserves name and input_data for simple string inputs."""
        original = [
            Task(name="Alpha", input_data={"goal": "Write docs"}),
            Task(name="Beta", input_data={"goal": "Fix bugs"}),
        ]
        dest = tmp_path / "rt.csv"

        loader = CSVTaskLoader()
        loader.save(original, dest)
        loaded = loader.load(dest)

        assert len(loaded) == len(original)
        for orig, reloaded in zip(original, loaded):
            assert reloaded.name == orig.name
            assert reloaded.input_data == orig.input_data

    def test_roundtrip_with_metadata(self, tmp_path: Path) -> None:
        """Metadata survives a CSV save/load cycle."""
        original = [Task(name="T", input_data={"q": "x"}, metadata={"region": "eu"})]
        dest = tmp_path / "meta_rt.csv"

        loader = CSVTaskLoader()
        loader.save(original, dest)
        loaded = loader.load(dest)

        assert loaded[0].metadata.get("region") == "eu"

    def test_roundtrip_custom_input_col(self, tmp_path: Path) -> None:
        """Custom input_col name is written and re-read consistently."""
        original = [Task(name="T", input_data={"goal": "custom"})]
        dest = tmp_path / "custom_col_rt.csv"
        loader = CSVTaskLoader(input_col="prompt")

        loader.save(original, dest)

        # Header must contain 'prompt', not 'input_data'.
        header_line = dest.read_text(encoding="utf-8").splitlines()[0]
        assert "prompt" in header_line
        assert "input_data" not in header_line

        loaded = loader.load(dest)
        assert loaded[0].input_data == {"goal": "custom"}

    def test_roundtrip_task_id_preserved(self, tmp_path: Path) -> None:
        """task_id is preserved across a CSV round-trip."""
        original = [Task(task_id="csv-id-99", name="T", input_data={})]
        dest = tmp_path / "ids.csv"

        loader = CSVTaskLoader()
        loader.save(original, dest)
        loaded = loader.load(dest)

        assert loaded[0].task_id == "csv-id-99"

    def test_roundtrip_multiple_metadata_cols(self, tmp_path: Path) -> None:
        """Multiple metadata columns all survive the round-trip."""
        original = [
            Task(
                name="T",
                input_data={"goal": "g"},
                metadata={"lang": "en", "domain": "science"},
            )
        ]
        dest = tmp_path / "multi_meta.csv"

        loader = CSVTaskLoader()
        loader.save(original, dest)
        loaded = loader.load(dest)

        assert loaded[0].metadata.get("lang") == "en"
        assert loaded[0].metadata.get("domain") == "science"


# ===========================================================================
# HFDatasetLoader
# ===========================================================================

# ---------------------------------------------------------------------------
# Fixtures: lightweight fake Dataset / DatasetDict that behave like the real
# objects but require no network access or extra install.
# ---------------------------------------------------------------------------


def _make_fake_dataset(rows: list[dict]) -> MagicMock:
    """Return a MagicMock that walks like a datasets.Dataset."""
    ds = MagicMock()
    ds.__iter__ = MagicMock(return_value=iter(rows))
    # Make isinstance(ds, DatasetDict) False and isinstance(ds, Dataset) True.
    ds.__class__.__name__ = "Dataset"
    return ds


def _make_fake_dataset_dict(splits: dict[str, list[dict]]) -> MagicMock:
    """Return a MagicMock that walks like a datasets.DatasetDict."""
    dd = MagicMock(spec=["__contains__", "__getitem__", "keys", "__iter__"])
    dd.__contains__ = lambda self, k: k in splits
    dd.__getitem__ = lambda self, k: _make_fake_dataset(splits[k])
    dd.keys.return_value = list(splits.keys())
    return dd


def _hf_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch 'datasets' into sys.modules so HFDatasetLoader.__init__ succeeds."""
    fake_hf = MagicMock()
    fake_hf.DatasetDict = MagicMock  # anything — isinstance check uses spec below
    monkeypatch.setitem(__import__("sys").modules, "datasets", fake_hf)


@pytest.fixture()
def fake_hf_module(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Inject a fake 'datasets' module and return it."""
    import sys
    fake = MagicMock()
    # DatasetDict needs to be a real class for isinstance() in _iter_dataset.
    class FakeDatasetDict(dict):  # noqa: N801
        pass
    fake.DatasetDict = FakeDatasetDict
    monkeypatch.setitem(sys.modules, "datasets", fake)
    return fake


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHFDatasetLoader:
    """Tests for HFDatasetLoader."""

    # -----------------------------------------------------------------------
    # Guard: missing extra raises a clear ImportError
    # -----------------------------------------------------------------------

    def test_import_error_when_datasets_not_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Constructing HFDatasetLoader without the [datasets] extra raises ImportError
        with an actionable install command in the message."""
        import sys
        # Remove datasets from sys.modules so the import inside __init__ fails.
        monkeypatch.delitem(sys.modules, "datasets", raising=False)
        # Also block importlib from finding the real package, if installed.
        import builtins
        real_import = builtins.__import__

        def _block(name: str, *args: object, **kwargs: object) -> object:
            if name == "datasets":
                raise ImportError("No module named 'datasets'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block)

        with pytest.raises(ImportError, match=r"tracelens\[datasets\]"):
            HFDatasetLoader()

    # -----------------------------------------------------------------------
    # load() — using pre-loaded Dataset objects (no network)
    # -----------------------------------------------------------------------

    def test_load_from_dataset_object(
        self, fake_hf_module: MagicMock
    ) -> None:
        """Passing a pre-loaded Dataset object converts rows to Tasks."""
        rows = [
            {"input": {"goal": "Write a haiku"}, "name": "Haiku"},
            {"input": {"goal": "Summarise article"}, "name": "Summary"},
        ]
        ds = _make_fake_dataset(rows)
        # Make isinstance(ds, fake_hf_module.DatasetDict) return False.
        fake_hf_module.DatasetDict = type("DatasetDict", (), {})

        loader = HFDatasetLoader()
        tasks = loader.load(ds)

        assert len(tasks) == 2
        assert tasks[0].name == "Haiku"
        assert tasks[0].input_data == {"goal": "Write a haiku"}
        assert tasks[1].name == "Summary"

    def test_load_scalar_input_wrapped(
        self, fake_hf_module: MagicMock
    ) -> None:
        """Scalar input values are wrapped as {'value': ...}."""
        rows = [{"input": "Plain string prompt", "name": "T"}]
        ds = _make_fake_dataset(rows)
        fake_hf_module.DatasetDict = type("DatasetDict", (), {})

        tasks = HFDatasetLoader().load(ds)

        assert tasks[0].input_data == {"value": "Plain string prompt"}

    def test_load_custom_input_field(
        self, fake_hf_module: MagicMock
    ) -> None:
        """Custom input_field is mapped to Task.input_data."""
        rows = [{"prompt": "Translate to French", "name": "Translation"}]
        ds = _make_fake_dataset(rows)
        fake_hf_module.DatasetDict = type("DatasetDict", (), {})

        tasks = HFDatasetLoader(input_field="prompt").load(ds)

        assert tasks[0].input_data == {"value": "Translate to French"}

    def test_load_metadata_fields_selected(
        self, fake_hf_module: MagicMock
    ) -> None:
        """Only requested metadata_fields end up in Task.metadata."""
        rows = [{
            "input": {"goal": "Do something"},
            "name": "T",
            "subject": "math",
            "level": "hard",
            "internal": "skip-me",
        }]
        ds = _make_fake_dataset(rows)
        fake_hf_module.DatasetDict = type("DatasetDict", (), {})

        tasks = HFDatasetLoader(
            metadata_fields=["subject", "level"]
        ).load(ds)

        assert tasks[0].metadata == {"subject": "math", "level": "hard"}
        assert "internal" not in tasks[0].metadata

    def test_load_metadata_all_by_default(
        self, fake_hf_module: MagicMock
    ) -> None:
        """When metadata_fields is None, all non-reserved, non-input keys go to metadata."""
        rows = [{
            "input": {"q": "x"},
            "name": "T",
            "source": "web",
        }]
        ds = _make_fake_dataset(rows)
        fake_hf_module.DatasetDict = type("DatasetDict", (), {})

        tasks = HFDatasetLoader().load(ds)

        assert tasks[0].metadata.get("source") == "web"

    def test_load_from_dataset_dict_with_split(
        self, fake_hf_module: MagicMock
    ) -> None:
        """A DatasetDict is resolved to the requested split."""
        rows = [{"input": {"goal": "Train task"}, "name": "Train T"}]

        class FakeDatasetDict:
            """Minimal DatasetDict stand-in."""
            def __contains__(self, item: str) -> bool:
                return item == "train"
            def __getitem__(self, key: str) -> MagicMock:
                return _make_fake_dataset(rows)
            def keys(self) -> list[str]:
                return ["train", "test"]

        fake_hf_module.DatasetDict = FakeDatasetDict
        dd = FakeDatasetDict()

        tasks = HFDatasetLoader().load(dd, split="train")

        assert len(tasks) == 1
        assert tasks[0].name == "Train T"

    def test_load_dataset_dict_missing_split_raises(
        self, fake_hf_module: MagicMock
    ) -> None:
        """Requesting a non-existent split from a DatasetDict raises ValueError."""
        class FakeDatasetDict:
            def __contains__(self, item: str) -> bool:
                return item == "train"
            def keys(self) -> list[str]:
                return ["train"]

        fake_hf_module.DatasetDict = FakeDatasetDict
        dd = FakeDatasetDict()

        with pytest.raises(ValueError, match="not found"):
            HFDatasetLoader().load(dd, split="test")

    def test_load_dataset_dict_no_split_raises(
        self, fake_hf_module: MagicMock
    ) -> None:
        """A DatasetDict without split= raises ValueError with a hint."""
        class FakeDatasetDict:
            def __contains__(self, item: str) -> bool:
                return False
            def keys(self) -> list[str]:
                return ["train", "test"]

        fake_hf_module.DatasetDict = FakeDatasetDict
        dd = FakeDatasetDict()

        with pytest.raises(ValueError, match="split="):
            HFDatasetLoader().load(dd)

    def test_load_from_hub_string(
        self, fake_hf_module: MagicMock
    ) -> None:
        """A Hub identifier string triggers load_dataset and returns Tasks."""
        rows = [{"input": {"question": "What is 2+2?"}, "name": "Math Q"}]
        fake_hf_module.load_dataset.return_value = _make_fake_dataset(rows)
        fake_hf_module.DatasetDict = type("DatasetDict", (), {})

        tasks = HFDatasetLoader().load("dummy/dataset", split="test")

        fake_hf_module.load_dataset.assert_called_once_with(
            "dummy/dataset", name=None, split="test"
        )
        assert len(tasks) == 1
        assert tasks[0].name == "Math Q"

    def test_load_from_path_object(
        self, fake_hf_module: MagicMock, tmp_path: Path
    ) -> None:
        """A pathlib.Path triggers load_dataset with the path as a string."""
        rows = [{"input": {"goal": "local"}, "name": "Local"}]
        fake_hf_module.load_dataset.return_value = _make_fake_dataset(rows)
        fake_hf_module.DatasetDict = type("DatasetDict", (), {})

        HFDatasetLoader().load(tmp_path / "my_ds", split="train")

        call_args = fake_hf_module.load_dataset.call_args
        assert isinstance(call_args.args[0], str)

    # -----------------------------------------------------------------------
    # save() — round-trip via Arrow save_to_disk
    # -----------------------------------------------------------------------

    def test_roundtrip_via_save_to_disk(
        self, fake_hf_module: MagicMock, tmp_path: Path
    ) -> None:
        """save() serialises tasks into Dataset.from_list and calls save_to_disk."""
        original = [
            Task(name="T1", input_data={"goal": "alpha"}, metadata={"src": "web"}),
            Task(name="T2", input_data={"goal": "beta"}),
        ]
        fake_ds = MagicMock()
        fake_hf_module.Dataset.from_list.return_value = fake_ds
        dest = tmp_path / "arrow_out"

        HFDatasetLoader().save(original, dest)

        fake_hf_module.Dataset.from_list.assert_called_once()
        records = fake_hf_module.Dataset.from_list.call_args.args[0]
        assert len(records) == 2
        # input_data should be renamed to the configured input_field.
        assert "input" in records[0]
        assert "input_data" not in records[0]
        assert records[0]["name"] == "T1"
        fake_ds.save_to_disk.assert_called_once_with(str(dest))
