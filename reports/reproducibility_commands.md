# Reproducibility Commands

## Activate environment

```bash
conda activate recomart
```

## Check repository and DVC state

```bash
git status
dvc status
```

## Restore DVC-tracked data and models if needed

```bash
dvc pull
```

## Run full Retailrocket Prefect pipeline

```bash
python -m orchestration.retailrocket_pipeline
```

## Schedule Retailrocket API ingestion every 30 minutes

Prefect 3 local deployment command:

```bash
prefect deploy orchestration/retailrocket_api_ingestion_flow.py:retailrocket_api_ingestion_flow \
  --name retailrocket-api-every-30-minutes \
  --interval 1800 \
  --pool default-agent-pool
```

The API ingestion flow defaults to mock mode unless `RECOMART_CATALOG_API_MOCK_MODE` is already set.

## Run Retailrocket training only

```bash
python -m src.training.train_retailrocket_popularity_recommender
python -m src.training.train_retailrocket_content_recommender
```

## Run Retailrocket inference

```bash
python -m src.serving.recommend_retailrocket --model popularity --top-k 5
python -m src.serving.recommend_retailrocket --model content_based --top-k 5
python -m src.serving.recommend_retailrocket --model popularity --user-id 1327109 --top-k 10
python -m src.serving.recommend_retailrocket --model content_based --demo-users 1327109,925350,839657 --top-k 5
```

## Check DuckDB warehouse

```bash
duckdb data/warehouse/recomart.duckdb "SELECT event_type, COUNT(*) FROM mart.retailrocket_interactions GROUP BY event_type;"
```

## Regenerate reports

```bash
python -m src.reporting.generate_assignment_evidence
python -m src.reporting.generate_model_comparison
python -m src.reporting.generate_final_report
python -m src.reporting.generate_assignment_pdf
```