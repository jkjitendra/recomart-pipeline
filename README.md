# RecoMart Retailrocket Recommendation Pipeline

RecoMart is a Retailrocket-only recommendation data management pipeline for Assignment I: End-to-End Data Management Pipeline for a Recommendation System. It ingests Retailrocket batch CSV files and Retailrocket item catalog metadata deltas from a REST/mock API, validates and prepares the data, builds a DuckDB warehouse and feature store, trains recommendation models, tracks experiments with MLflow, orchestrates the workflow with Prefect, and generates assignment evidence and a consolidated PDF report.

## Table of Contents

- [Business Problem](#business-problem)
- [Dataset Used](#dataset-used)
- [Final Architecture](#final-architecture)
- [Folder Structure](#folder-structure)
- [Tools Used](#tools-used)
- [Environment Setup](#environment-setup)
- [DVC Artifacts](#dvc-artifacts)
- [Run the Pipeline](#run-the-pipeline)
- [API Ingestion](#api-ingestion)
- [Training and Inference](#training-and-inference)
- [MLflow Tracking](#mlflow-tracking)
- [Reports and Evidence](#reports-and-evidence)
- [Submission Artifacts](#submission-artifacts)

## Business Problem

RecoMart needs a reproducible data pipeline that turns e-commerce user behavior and item metadata into recommendation features and model outputs. The pipeline supports product recommendation use cases such as ranking candidate items for active users, evaluating recommendation quality, and preserving data/model lineage for reproducibility.

## Dataset Used

The project uses Retailrocket data only:

- Batch interaction source: `data/external/retailrocket/events.csv`
- Batch item property source: `data/external/retailrocket/item_properties_part1.csv`
- REST/mock API source archive: `data/external/retailrocket/item_properties_part2.csv`
- Batch category hierarchy source: `data/external/retailrocket/category_tree.csv`

The REST API represents an external near-real-time item catalog metadata delta feed. The teammate-hosted API serves non-repeated records from `item_properties_part2.csv` and accepts a `count` query parameter controlled by `RECOMART_CATALOG_API_PAGE_SIZE`. Mock mode remains the default for reproducible local runs and uses cursor state against the local source archive.

## Final Architecture

```text
data/external/retailrocket/
  -> batch ingestion + REST/mock API ingestion
data/raw/source=<source_name>/type=<data_type>/ingestion_timestamp=<YYYYMMDD_HHMMSS>/
  -> raw validation
data/staged/source=retailrocket/
  -> staged and curated validation
data/curated/source=retailrocket/
  -> DuckDB warehouse
data/warehouse/recomart.duckdb
  -> feature tables and feature registry
data/features/source=retailrocket/
  -> model training + MLflow tracking
models/retailrocket/
  -> inference, reports, PDF, and Prefect evidence
```

Raw ingestion uses this partition convention:

```text
data/raw/source=<source_name>/type=<data_type>/ingestion_timestamp=<YYYYMMDD_HHMMSS>/
```

Implemented raw paths:

```text
data/raw/source=retailrocket_batch/type=events/ingestion_timestamp=<YYYYMMDD_HHMMSS>/
data/raw/source=retailrocket_batch/type=item_properties_part1/ingestion_timestamp=<YYYYMMDD_HHMMSS>/
data/raw/source=retailrocket_batch/type=category_tree/ingestion_timestamp=<YYYYMMDD_HHMMSS>/
data/raw/source=retailrocket_api/type=item_properties_delta/ingestion_timestamp=<YYYYMMDD_HHMMSS>/
```

## Folder Structure

```text
configs/          Retailrocket feature registry configuration
data/external/    DVC-tracked Retailrocket source archive
data/raw/         DVC-tracked immutable raw ingestion snapshots
data/staged/      DVC-tracked typed Retailrocket Parquet datasets
data/curated/     DVC-tracked EDA/modeling-ready analytical datasets
data/features/    DVC-tracked Retailrocket user, item, and user-item feature tables
data/warehouse/   Generated DuckDB warehouse, ignored by Git
logs/             Ingestion and pipeline logs
models/           DVC-tracked Retailrocket model artifacts
orchestration/    Prefect full pipeline and API ingestion flows
reports/          CSV evidence, plots, screenshots, markdown report, and PDF report
sql/              Retailrocket DuckDB SQL scripts
src/              Python modules for ingestion, validation, preparation, transformation, training, serving, and reporting
```

## Tools Used

| Tool | Purpose |
|---|---|
| Python | Pipeline implementation and command entry points |
| pandas | CSV/Parquet processing, validation summaries, report tables |
| DuckDB | Local SQL warehouse and transformation layer |
| Parquet | Efficient staged, curated, and feature storage |
| DVC | Versioning large data and model artifacts without committing payloads to Git |
| MLflow | Tracking model parameters, metrics, run IDs, and artifacts |
| Prefect | Local orchestration for the full Retailrocket pipeline and scheduled API ingestion |
| ReportLab | PDF report generation |
| FastAPI/mock API | Demo-compatible REST API source for item property deltas |

## Environment Setup

Use the existing conda environment:

```bash
conda activate recomart
```

Or run commands without activating the shell:

```bash
conda run -n recomart <command>
```

## DVC Artifacts

Restore DVC-tracked artifacts if a DVC remote is configured:

```bash
dvc pull
```

Check versioning state:

```bash
git status
dvc status
```

DVC metadata tracks large generated artifacts:

- `data/external/retailrocket.dvc`
- `data/raw/source=retailrocket_batch.dvc`
- `data/raw/source=retailrocket_api.dvc`
- `data/staged/source=retailrocket.dvc`
- `data/curated/source=retailrocket.dvc`
- `data/features/source=retailrocket.dvc`
- `models/retailrocket.dvc`

Generated heavy data/model payloads are DVC-tracked and are not committed directly to Git.

## Run the Pipeline

Run the full Retailrocket Prefect pipeline:

```bash
python -m orchestration.retailrocket_pipeline
```

The flow runs batch ingestion, API/mock delta ingestion, raw validation, staged/curated preparation, staged/curated validation, DuckDB loading, feature building, feature retrieval, model training, inference, model comparison, evidence generation, final markdown report generation, and PDF generation.

## API Ingestion

Run one Retailrocket API/mock ingestion step:

```bash
RECOMART_CATALOG_API_MOCK_MODE=true python -m src.ingestion.ingest_retailrocket_catalog_api
```

Mock mode is the reproducible default for the full local pipeline. It reads from `data/external/retailrocket/item_properties_part2.csv`, fetches 10 records per run unless `RECOMART_CATALOG_API_PAGE_SIZE` is set, and persists local cursor state.

Run against the teammate-hosted real API:

```bash
RECOMART_CATALOG_API_MOCK_MODE=false \
RECOMART_CATALOG_API_BASE_URL="<teammate-api-base-url>" \
RECOMART_CATALOG_API_ENDPOINT="/items" \
RECOMART_CATALOG_API_PAGE_SIZE=50 \
python -m src.ingestion.ingest_retailrocket_catalog_api
```

The real API response uses `data`, `success`, `count`, `total_rows`, and `unread_rows`. The ingestion client sends `?count=<page_size>`, normalizes rows to `timestamp`, `itemid`, `property`, `value`, `source_system`, and `ingestion_timestamp`, and writes immutable raw snapshots under:

```text
data/raw/source=retailrocket_api/type=item_properties_delta/ingestion_timestamp=<YYYYMMDD_HHMMSS>/
```

API configuration variables:

```text
RECOMART_CATALOG_API_BASE_URL
RECOMART_CATALOG_API_ENDPOINT
RECOMART_CATALOG_API_TIMEOUT_SEC
RECOMART_CATALOG_API_PAGE_SIZE
RECOMART_CATALOG_API_AUTH_TOKEN
RECOMART_CATALOG_API_STATE_FILE
RECOMART_CATALOG_API_MOCK_MODE
```

Run the optional local mock API server:

```bash
uvicorn src.ingestion.mock_retailrocket_catalog_api:app --host 127.0.0.1 --port 8000
```

Run the ingestion client against the mock server:

```bash
RECOMART_CATALOG_API_BASE_URL=http://127.0.0.1:8000 python -m src.ingestion.ingest_retailrocket_catalog_api
```

Schedule API ingestion every 30 minutes with Prefect 3:

```bash
prefect deploy orchestration/retailrocket_api_ingestion_flow.py:retailrocket_api_ingestion_flow \
  --name retailrocket-api-every-30-minutes \
  --interval 1800 \
  --pool default-agent-pool
```

The API ingestion flow defaults to mock mode unless `RECOMART_CATALOG_API_MOCK_MODE` is already set.

## Training and Inference

Train the Retailrocket models:

```bash
python -m src.training.train_retailrocket_popularity_recommender
python -m src.training.train_retailrocket_content_recommender
```

Run inference for both models:

```bash
python -m src.serving.recommend_retailrocket --model popularity --top-k 5
python -m src.serving.recommend_retailrocket --model content_based --top-k 5
```

Run inference for selected users:

```bash
python -m src.serving.recommend_retailrocket --model popularity --user-id 1327109 --top-k 10
python -m src.serving.recommend_retailrocket --model content_based --demo-users 1327109,925350,839657 --top-k 5
```

## MLflow Tracking

The models log to the MLflow experiment:

```text
retailrocket_recommendation_models
```

Open the MLflow UI with the local tracking backend:

```bash
mlflow ui --backend-store-uri sqlite:///$(pwd)/mlflow.db --default-artifact-root $(pwd)/mlruns --port 5001
```

If runs were created with the default local file store instead of SQLite, open:

```bash
mlflow ui --backend-store-uri file:$(pwd)/mlruns --port 5001
```

MLflow screenshot evidence is stored in `reports/screenshots/`.

## Reports and Evidence

Regenerate evidence, markdown report, and PDF:

```bash
python -m src.reporting.generate_assignment_evidence
python -m src.reporting.generate_model_comparison
python -m src.reporting.generate_final_report
python -m src.reporting.generate_assignment_pdf
```

Important evidence files:

- `reports/recomart_assignment_report.pdf`
- `reports/final_project_report.md`
- `reports/reproducibility_commands.md`
- `reports/orchestration_retailrocket_pipeline_summary.csv`
- `reports/feature_metadata_documentation.csv`
- `reports/feature_store_retrieval_demo.csv`
- `reports/retailrocket_model_comparison.csv`
- `reports/submission_checklist.md`
- `reports/video_walkthrough_script.md`

## Submission Artifacts

Expected submission materials:

- Source code in this repository
- Consolidated PDF report: `reports/recomart_assignment_report.pdf`
- Reproducibility commands: `reports/reproducibility_commands.md`
- Video walkthrough script: `reports/video_walkthrough_script.md`
- DVC metadata for large data/model artifacts
- MLflow screenshot evidence under `reports/screenshots/`
- Prefect orchestration evidence under `reports/orchestration_retailrocket_pipeline_summary.csv`

Final ZIP packaging is pending explicit approval.
