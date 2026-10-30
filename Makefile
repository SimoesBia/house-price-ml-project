.PHONY: install install-dev eda train tune serve test lint format docker-build docker-run

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

eda:
	python main.py eda

train:
	python main.py train

tune:
	python main.py train --tune

serve:
	python main.py serve

test:
	pytest --cov=src/house_price --cov-report=term-missing

lint:
	ruff check .

format:
	ruff format .

docker-build:
	docker build -t house-price-api .

docker-run:
	docker run --rm -p 8000:8000 -v $(PWD)/models:/app/models house-price-api
