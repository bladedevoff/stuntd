# Contributing

Install the project and its development extras with `make sync`, or `pip install -e ".[dev]"`.

- `make test` runs the suite: `pytest`.
- `make lint` checks style: `ruff format --check .` and `ruff check .`.
- `make typecheck` runs `mypy stuntd` and `pyright stuntd`.
- `make format` rewrites the tree: `ruff format .` and `ruff check --fix .`.
- `make check` is lint, typecheck and test together, the same commands CI runs.
- `make mutate` runs `mutmut run` over `stuntd/train/metrics.py` and `stuntd/train/dataset.py`,
  then `mutmut results`; it is slow and not part of CI, so run it by hand before trusting a change
  to those modules.

Training lives behind the `train` extra (`pip install -e ".[dev,train]"`); a real run is
`pytest -m slow` with `STUNTD_LAYA_MODEL=<local checkpoint directory>`. Serving needs that same
extra whenever a head answers: for sites in shadow or live mode, and for local Jev (`[jev]
upstream` left empty), where `stuntd.serve.decider` answers from the base checkpoint in the proxy
process.

`pre-commit install` puts ruff and mypy on the commit hook; `pre-commit run --all-files` runs them by hand.

A pull request keeps to one topic, leaves `make check` green, and covers new behaviour with a test.
Public symbols carry a one-line docstring. Write a comment only where the code cannot explain
itself, one plain sentence, no prefixes. Tests carry no comments.
