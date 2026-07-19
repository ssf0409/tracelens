# Loading Tasks from External Sources

TraceLens evaluations are built around [`Task`][tracelens.Task] objects, but your data
almost certainly already lives in a spreadsheet, a log file, or a Hugging Face dataset.
The loaders in this page let you point TraceLens at those files directly — no conversion
step required.

## Overview

| Loader | Format | Extra required |
|---|---|---|
| `JSONTaskLoader` | JSON file(s) / directory | — (core) |
| `JSONLTaskLoader` | Newline-delimited JSON (`.jsonl`) | — (stdlib) |
| `CSVTaskLoader` | Comma-separated values (`.csv`) | — (stdlib) |
| `HFDatasetLoader` | Hugging Face Hub / saved `Dataset` | `tracelens[datasets]` |

All loaders implement the same [`TaskLoader`][tracelens.core.task.TaskLoader] ABC, so
their `load()` and `save()` result contracts remain consistent:

```python
loader = CSVTaskLoader(input_field="prompt")   # or any other loader
tasks  = loader.load("eval_data.csv")
eval_set = EvalSet(name="My Suite", tasks=tasks)
```

---

## JSONLTaskLoader

Read one JSON object per line from a `.jsonl` file.

### Installation

No extra dependencies — ships with the `tracelens` core package.

### Basic usage

```python
from tracelens import JSONLTaskLoader

tasks = JSONLTaskLoader().load("evals.jsonl")
```

Each line must be a valid JSON object with at least an `"input"` key:

```jsonl
{"input": {"goal": "Write a haiku about autumn"}, "name": "Haiku task"}
{"input": {"goal": "Summarise the paper"}, "name": "Summarisation", "difficulty": "hard"}
```

### Custom field names

If your file uses a different column name for the input, pass `input_field`:

```python
tasks = JSONLTaskLoader(input_field="prompt").load("my_prompts.jsonl")
```

Limit which extra keys end up in `Task.metadata` with `metadata_fields`:

```python
tasks = JSONLTaskLoader(
    input_field="prompt",
    metadata_fields=["subject", "source"],
).load("benchmarks.jsonl")
```

`metadata_fields` may select only foreign keys. Native `Task` fields such as
`difficulty` and `category` always map to their corresponding Task attributes,
so a field never has two meanings. It filters flat foreign keys only; an embedded
canonical `metadata` object is already normalized and is therefore loaded intact.

### Directory loading

Pass a directory path and every `*.jsonl` file inside (including subdirectories)
is loaded in sorted order:

```python
tasks = JSONLTaskLoader().load("data/evals/")
```

### Saving

`save()` writes one JSON object per line, renaming `input_data` back to your
configured `input_field` so round-trips work cleanly:

```python
loader = JSONLTaskLoader()
loader.save(tasks, "output.jsonl")
tasks2 = loader.load("output.jsonl")  # identical to `tasks`
```

---

## CSVTaskLoader

Read a `.csv` file via the standard-library [`csv`](https://docs.python.org/3/library/csv.html)
module.

### Installation

No extra dependencies — ships with the `tracelens` core package.

### Basic usage

```python
from tracelens import CSVTaskLoader

tasks = CSVTaskLoader().load("evals.csv")
```

A minimal CSV looks like:

```csv
input,name,difficulty
"What is 2 + 2?",Math addition,easy
"Summarise the abstract",Summarisation,hard
```

### Custom column names

Map a different column to `Task.input_data` with `input_field`, and select
which columns become `Task.metadata` with `metadata_fields`:

```python
tasks = CSVTaskLoader(
    input_field="prompt",
    metadata_fields=["subject", "source"],
).load("my_data.csv")
```

When `metadata_fields` is omitted, **all** columns that are not `input_field` and
not one of the reserved Task fields (`name`, `difficulty`, `category`, …) are
collected into `Task.metadata` automatically.

The configured input field is required. A missing JSONL field or CSV column is
reported as an error instead of producing an empty task. It must not reuse a
native `Task` field name such as `name` or `metadata`.

CSV headers must be unique and non-blank, and every row must have no more values
than the header. Malformed source structure fails with the file and line location
instead of silently dropping data.

### JSON-encoded cells

Cells that contain a valid JSON object or array are parsed automatically, so
you can store structured input data in a single CSV column:

```csv
input,name
"{""goal"": ""Write a haiku"", ""lang"": ""EN""}",Haiku task
```

### Directory loading

```python
tasks = CSVTaskLoader().load("data/csvs/")  # globs **/*.csv recursively
```

### Saving

```python
loader = CSVTaskLoader()
loader.save(tasks, "output.csv")
tasks2 = loader.load("output.csv")
```

TraceLens writes structured Task fields as JSON and uses one canonical JSON
`metadata` column. This preserves arbitrary JSON-compatible metadata,
including keys such as `name` or `input` that would collide with ordinary CSV
columns. External CSV files may instead use flat extra columns as metadata,
but a row cannot combine a `metadata` column with flat metadata columns: that
ambiguous representation is rejected.

---

## Design policy: wrap parsers, own only the mapping

TraceLens loaders never implement file-format parsing. `JSONLTaskLoader` and
`CSVTaskLoader` delegate to the stdlib `json` and `csv` modules — the
maintained parsers for those formats — and own only the row-to-`Task`
mapping (which columns are Task fields, what gets JSON-parsed vs kept as
text, how metadata is collected). The same rule applies to future sources:
a Parquet loader would wrap `pyarrow`, a database loader would wrap its
driver, and hosted-dataset sources wrap their ecosystem client. Optional
integrations remain isolated from core imports and reuse the same record
mapping contract as local loaders.

## HFDatasetLoader

Install the optional dependency:

```bash
pip install "tracelens[datasets]"
```

Hub datasets require an explicit split. Pin `revision` to a commit SHA for
reproducible CI evaluations:

```python
from tracelens.loaders import HFDatasetLoader

loader = HFDatasetLoader(
    input_field="question",
    metadata_fields=["subject"],
    config_name="all",
    split="test",
    revision="<dataset-commit-sha>",
)
tasks = loader.load("cais/mmlu")
```

String sources are always treated as Hub dataset identifiers. A local
dataset previously written by `datasets.Dataset.save_to_disk()` uses an
explicit `Path`:

```python
from pathlib import Path

loader.save(tasks, Path("eval-data/mmlu"))
reloaded = loader.load(Path("eval-data/mmlu"))
```

`HFDatasetLoader` does not stream or push datasets to the Hub. Its `load()`
contract returns a materialized `list[Task]`, and `save()` writes only a local
Hugging Face dataset directory. Saving an empty Task list is rejected because
Hugging Face cannot infer a reloadable dataset schema without any rows.
