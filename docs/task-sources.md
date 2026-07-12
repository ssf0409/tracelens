# Loading Tasks from External Sources

TraceLens evaluations are built around :class:`~tracelens.core.task.Task` objects, but your
data almost certainly already lives in a spreadsheet, a log file, or a HuggingFace
dataset. The loaders in this page let you point TraceLens at those files directly —
no conversion step required.

## Overview

| Loader | Format | Extra required |
|---|---|---|
| `JSONTaskLoader` | JSON file(s) / directory | — (core) |
| `JSONLTaskLoader` | Newline-delimited JSON (`.jsonl`) | — (stdlib) |
| `CSVTaskLoader` | Comma-separated values (`.csv`) | — (stdlib) |
| `HFDatasetLoader` | HuggingFace `datasets` | `tracelens[datasets]` |

All loaders implement the same [`TaskLoader`][tracelens.core.task.TaskLoader] ABC so
they are interchangeable in your code:

```python
loader = CSVTaskLoader(input_col="prompt")   # or any other loader
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
    metadata_fields=["subject", "difficulty"],
).load("benchmarks.jsonl")
```

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

Map a different column to `Task.input_data` with `input_col`, and select
which columns become `Task.metadata` with `metadata_cols`:

```python
tasks = CSVTaskLoader(
    input_col="prompt",
    metadata_cols=["category", "source"],
).load("my_data.csv")
```

When `metadata_cols` is omitted, **all** columns that are not `input_col` and
not one of the reserved Task fields (`name`, `difficulty`, `category`, …) are
collected into `Task.metadata` automatically.

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

---

## HFDatasetLoader

Load tasks directly from a [HuggingFace `datasets`](https://huggingface.co/docs/datasets/)
`Dataset` or `DatasetDict` — including any dataset on the Hub.

### Installation

```bash
pip install "tracelens[datasets]"
# or
uv pip install "tracelens[datasets]"
```

If you try to use `HFDatasetLoader` without the extra installed, you'll get a
clear error immediately:

```
ImportError: HFDatasetLoader requires the 'datasets' package.
Install it with:

    pip install "tracelens[datasets]"
```

### Load from the HuggingFace Hub

```python
from tracelens import HFDatasetLoader

tasks = HFDatasetLoader(
    input_field="question",
    metadata_fields=["subject", "level"],
).load("cais/mmlu", split="test")
```

### Load from a local Arrow/Parquet dataset

```python
tasks = HFDatasetLoader().load("./my_local_dataset", split="train")
```

### Pass a pre-loaded Dataset object

Useful in tests or when you've already done preprocessing:

```python
import datasets as hf

ds = hf.Dataset.from_list([
    {"input": {"goal": "Write a haiku"}, "name": "Haiku"},
    {"input": {"goal": "Classify sentiment"}, "name": "Sentiment"},
])

tasks = HFDatasetLoader().load(ds)
```

### Working with DatasetDict (multiple splits)

When the source is a `DatasetDict` (the common Hub format), you must specify
which split to use:

```python
# Fails clearly if split= is omitted:
tasks = HFDatasetLoader().load("squad", split="validation")
```

### Custom field names

```python
tasks = HFDatasetLoader(
    input_field="context",
    metadata_fields=["id", "title"],
).load("squad", split="validation")
```

### Saving

`save()` writes an Arrow dataset directory that can be reloaded with
`datasets.load_from_disk()`:

```python
loader = HFDatasetLoader()
loader.save(tasks, "arrow_output/")
# reload later: datasets.load_from_disk("arrow_output/")
```

---

## Reserved column / field names

The following names are recognised as Task model fields and are forwarded
**directly** to the `Task` constructor rather than being placed into
`Task.metadata`. This is true for all three loaders:

| Name | Task field |
|---|---|
| `task_id` | `Task.task_id` |
| `name` | `Task.name` |
| `description` | `Task.description` |
| `tags` | `Task.tags` |
| `difficulty` | `Task.difficulty` |
| `category` | `Task.category` |
| `timeout_seconds` | `Task.timeout_seconds` |
| `max_retries` | `Task.max_retries` |
| `expectation` | `Task.expectation` |
| `metadata` | `Task.metadata` (used verbatim on round-trip) |

---

## Choosing a loader

| Situation | Recommended loader |
|---|---|
| You have `.jsonl` logs from your agent runs | `JSONLTaskLoader` |
| You have a spreadsheet / exported CSV | `CSVTaskLoader` |
| You want to evaluate against a published benchmark | `HFDatasetLoader` |
| You created tasks programmatically with TraceLens | `JSONTaskLoader` |
