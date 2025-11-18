.PHONY: install format lint test train

install:
	pip install -e .[dev]

format:
	ruff check src tests --fix
	black src tests

lint:
	ruff check src tests
	mypy src

test:
	pytest -q

train:
	python scripts/train_gp_ssm.py --config configs/default.yaml
