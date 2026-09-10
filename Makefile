install:
	pip install -r requirements.txt

ingest:
	python src/ingestion/ingest.py

validate:
	python src/validation/validate.py

features:
	python src/features/build_features.py

test:
	pytest

	