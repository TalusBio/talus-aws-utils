.venv:
	poetry install
	pre-commit install

format: .venv
	poetry run isort .
	poetry run black .
	poetry run flake8 .

test: .venv
	poetry run python -m pytest --durations=0 -s $(FILTER)

pre-commit: .venv
	pre-commit run --all-files
