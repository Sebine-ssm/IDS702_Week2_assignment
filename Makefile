install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black *.py

test:
	pytest --cov=analysis --cov-report=term-missing

lint:
	flake8 --ignore=analysis.py

clean:
	rm -rf __pycache__ .pytest_cache .coverage

all:
	install format lint test clean 