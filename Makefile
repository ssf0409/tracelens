# Single-entry verification gate. `make verify` runs everything CI runs
# locally: lock check -> lint -> typecheck -> tests (with the coverage floor
# CI enforces).

.PHONY: test lint typecheck coverage format lockcheck verify docs docs-serve

test:
	uv run --frozen pytest -q

coverage:
	uv run --frozen pytest -q --cov=tracelens --cov-report=term-missing --cov-fail-under=90

lint:
	uv run --frozen ruff check src/ tests/ examples/ benchmarks/high-stakes-autonomous

format:
	uv run --frozen ruff format src/ tests/ examples/

typecheck:
	uv run --frozen --extra dev mypy src/tracelens/

lockcheck:
	uv lock --check

verify: lockcheck lint typecheck coverage
	@echo "✓ verify passed"

# Build the documentation site with the same strict link/reference checks CI runs.
docs:
	uv run --extra docs mkdocs build --strict

# Live-reloading local preview at http://127.0.0.1:8000
docs-serve:
	uv run --extra docs mkdocs serve
