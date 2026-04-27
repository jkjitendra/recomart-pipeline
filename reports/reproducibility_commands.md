# Reproducibility Commands

## Activate Environment

```bash
conda activate recomart
```

## Check Repository and DVC State

```bash
git status
dvc status
```

## Restore DVC-Tracked Data and Models

```bash
dvc pull
```

## Run Full Retailrocket Prefect Pipeline

```bash
python -m orchestration.retailrocket_pipeline
```

## Run API Ingestion Only

```bash
RECOMART_CATALOG_API_MOCK_MODE=true python -m src.ingestion.ingest_retailrocket_catalog_api
```

## Schedule API Ingestion Every 30 Minutes

```bash
prefect deploy orchestration/retailrocket_api_ingestion_flow.py:retailrocket_api_ingestion_flow \
  --name retailrocket-api-every-30-minutes \
  --interval 1800 \
  --pool default-agent-pool
```

## Run Retailrocket Training Only

```bash
python -m src.training.train_retailrocket_popularity_recommender
python -m src.training.train_retailrocket_content_recommender
```

## Run Retailrocket Inference

```bash
python -m src.serving.recommend_retailrocket --model popularity --top-k 5
python -m src.serving.recommend_retailrocket --model content_based --top-k 5
```

## Open MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///$(pwd)/mlflow.db --default-artifact-root $(pwd)/mlruns --port 5001
```

If local runs are stored in the default file backend, use:

```bash
mlflow ui --backend-store-uri file:$(pwd)/mlruns --port 5001
```

## Regenerate Evidence and Reports

```bash
python -m src.reporting.generate_assignment_evidence
python -m src.reporting.generate_model_comparison
python -m src.reporting.generate_final_report
python -m src.reporting.generate_assignment_pdf
```

## Check DuckDB Warehouse

```bash
duckdb data/warehouse/recomart.duckdb "SELECT event_type, COUNT(*) FROM mart.retailrocket_interactions GROUP BY event_type;"
```
