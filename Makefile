.venv:
	poetry install

format: .venv
	poetry run isort .
	poetry run black .
	poetry run flake8 .

test: .venv
	poetry run python -m pytest --durations=0 -s $(FILTER)
