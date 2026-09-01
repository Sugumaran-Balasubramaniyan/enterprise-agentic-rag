.PHONY: install test benchmark eval run ui lint docker-up docker-down

install:
	pip install -r requirements.txt

test:
	pytest -v --tb=short

benchmark:
	python3 benchmarks/latency_benchmark.py

eval:
	python3 benchmarks/grounding_eval.py

run:
	uvicorn app.main:app --reload --port 8000

ui:
	streamlit run streamlit_app.py --server.port 8501

lint:
	ruff check --select=E9,F63,F7,F82 .

docker-up:
	docker compose up -d

docker-down:
	docker compose down
