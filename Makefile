.venv:
	poetry install

format: .venv
	poetry run isort .
	poetry run black .

test: .venv
	poetry run python -m pytest --durations=0 -s $(FILTER)