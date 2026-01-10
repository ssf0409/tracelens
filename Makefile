.PHONY: help test test-local test-docker test-coverage lint type-check build clean dev

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Local development
test-local:  ## Run tests locally
	pytest tests/ -v --tb=short

test-coverage-local:  ## Run tests with coverage locally
	pytest tests/ -v --cov=agent_eval --cov-report=html --cov-report=term-missing

lint-local:  ## Run linting locally
	ruff check src/ tests/

type-check-local:  ## Run type checking locally
	mypy src/agent_eval/ --ignore-missing-imports

# Docker-based testing (no local dependencies required)
build:  ## Build Docker image
	docker compose build

test:  ## Run tests in Docker
	docker compose run --rm test

test-coverage:  ## Run tests with coverage in Docker
	docker compose run --rm test-coverage

lint:  ## Run linting in Docker
	docker compose run --rm lint

type-check:  ## Run type checking in Docker
	docker compose run --rm type-check

dev:  ## Start interactive development shell in Docker
	docker compose run --rm dev

# All checks (Docker)
check-all:  ## Run all checks in Docker (test, lint, type-check)
	docker compose run --rm test
	docker compose run --rm lint
	docker compose run --rm type-check

# All checks (local)
check-all-local:  ## Run all checks locally
	pytest tests/ -v --tb=short
	ruff check src/ tests/
	mypy src/agent_eval/ --ignore-missing-imports

# Cleanup
clean:  ## Clean up Docker resources and artifacts
	docker compose down --rmi local --volumes --remove-orphans
	rm -rf .pytest_cache .mypy_cache coverage/ htmlcov/ *.egg-info dist/ build/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
