.PHONY: sync format lint typecheck test check mutate

sync:
	pip install -e ".[dev]"

format:
	ruff format .
	ruff check --fix .

lint:
	ruff format --check .
	ruff check .

typecheck:
	mypy stuntd
	pyright stuntd

test:
	pytest

check: lint typecheck test

mutate:
	mutmut run
	mutmut results
