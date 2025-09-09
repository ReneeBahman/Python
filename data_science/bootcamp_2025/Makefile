.PHONY: init lab fmt lint

init:
	python -m venv .venv && \
	(. .venv/Scripts/activate || . .venv/bin/activate; \
	pip install --upgrade pip; \
	pip install -r env/requirements.txt -r env/requirements_dev.txt; \
	python -m ipykernel install --user --name "bootcamp_2025" --display-name "Python (bootcamp_2025)")

lab:
	(. .venv/Scripts/activate || . .venv/bin/activate; jupyter lab)

fmt:
	(. .venv/Scripts/activate || . .venv/bin/activate; black . && isort .)

lint:
	(. .venv/Scripts/activate || . .venv/bin/activate; ruff .)
