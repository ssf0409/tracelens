"""Round-trip tests for CSVTaskLoader and JSONLTaskLoader."""

import csv
import json
from pathlib import Path

import pytest

from tracelens.core.task import Task
from tracelens.loaders import CSVTaskLoader, JSONLTaskLoader

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


class TestCSVReservedColumnFidelity:
    """Free-text reserved columns must survive verbatim: a name or
    description that happens to be valid JSON ("true", "123") must not be
    coerced into a bool/number, while structured columns (tags,
    expectation, timeout_seconds) are parsed."""

    def test_json_looking_text_columns_stay_strings(self, tmp_path: Path) -> None:
        _write_csv(
            tmp_path / "t.csv",
            [{
                "task_id": "001",
                "name": "123",
                "description": "true",
                "category": "null",
                "input": '{"q": "x"}',
            }],
            ["task_id", "name", "description", "category", "input"],
        )

        task = CSVTaskLoader().load(tmp_path / "t.csv")[0]

        assert task.task_id == "001"
        assert task.name == "123"
        assert task.description == "true"
        assert task.category == "null"

    def test_structured_columns_are_parsed(self, tmp_path: Path) -> None:
        _write_csv(
            tmp_path / "t.csv",
            [{
                "name": "n",
                "tags": '["smoke", "regression"]',
                "timeout_seconds": "42.5",
                "input": '{"q": "x"}',
            }],
            ["name", "tags", "timeout_seconds", "input"],
        )

        task = CSVTaskLoader().load(tmp_path / "t.csv")[0]

        assert task.tags == ["smoke", "regression"]
        assert task.timeout_seconds == 42.5

    def test_save_load_round_trip_preserves_tags_and_expectation(
        self, tmp_path: Path
    ) -> None:
        from tracelens.core.task import TaskExpectation

        original = Task(
            task_id="rt-1",
            name="round trip",
            description="d",
            input_data={"q": "x"},
            tags=["a", "b"],
            expectation=TaskExpectation(expected_output={"answer": "42"}),
        )

        loader = CSVTaskLoader()
        loader.save([original], tmp_path / "out.csv")
        loaded = loader.load(tmp_path / "out.csv")[0]

        assert loaded.tags == ["a", "b"]
        assert loaded.expectation is not None
        assert loaded.expectation.expected_output == {"answer": "42"}
