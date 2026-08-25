.PHONY: install test lint run docker-up docker-down

install:
	pip install -r requirements.txt

test:
	pytest -v

run:
	uvicorn app.main:app --reload --port 8000

docker-up:
	docker compose up -d

docker-down:
	docker compose down
