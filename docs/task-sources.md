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
| custom (recipe below) | HuggingFace `datasets` | `pip install datasets` |

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

## HuggingFace datasets (recipe)

TraceLens deliberately does not ship a HuggingFace loader: hosted-dataset
integrations start as recipes until enough downstream projects need a shared
abstraction (see ROADMAP non-goals). Loading from `datasets` is a few lines
against the same `Task` model:

```python
from datasets import load_dataset  # pip install datasets

from tracelens import Task

def tasks_from_hf(name: str, split: str, input_field: str = "question") -> list[Task]:
    rows = load_dataset(name, split=split)  # explicit split: no DatasetDict ambiguity
    return [
        Task(
            name=str(row[input_field])[:80],
            description=f"{name}:{split} row {i}",
            input_data={input_field: row[input_field]},
            metadata={k: v for k, v in row.items() if k != input_field},
        )
        for i, row in enumerate(rows)
    ]
```

Pin the dataset revision (`load_dataset(..., revision=...)`) if you need the
eval set to be reproducible, and save the result with `JSONLTaskLoader` so
reruns don't depend on the Hub.
