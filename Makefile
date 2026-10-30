.PHONY: app debug down test lint qa

app:
	COMPOSE_PROFILES=app docker compose up --build

debug:
	COMPOSE_PROFILES=debug docker compose up --build

down:
	docker compose down

test:
	.venv/bin/python -m pytest tests/ -v

lint:
	.venv/bin/python -m ruff check .
	.venv/bin/python -m ty check

qa: test lint

