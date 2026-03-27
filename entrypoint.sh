#!/bin/sh
set -e

echo "==> Ingesting sample documents..."
python -m scripts.ingest_sample_docs --dir ./data/sample_docs --strategy semantic_sections

echo "==> Starting API server..."
exec uvicorn src.api.app:create_app --factory --host 0.0.0.0 --port 8000
