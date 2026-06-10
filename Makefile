# Single-entry verification gate. `make verify` runs everything CI runs
# locally: lint -> typecheck -> tests (with the coverage floor CI enforces).

.PHONY: test lint typecheck coverage format verify

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

verify: lint typecheck coverage
	@echo "✓ verify passed"
