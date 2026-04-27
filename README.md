# RecoMart Recommendation Data Pipeline

RecoMart is a reproducible data management pipeline for recommendation modeling on the Retailrocket dataset. The project ingests Retailrocket batch CSV data and supports a Retailrocket catalog metadata delta REST API source.

## Project Scope

RecoMart uses Retailrocket data to build a local ML data pipeline:

- Batch source archive: `data/external/retailrocket/events.csv`
- Batch source archive: `data/external/retailrocket/item_properties_part1.csv`
- Batch source archive: `data/external/retailrocket/item_properties_part2.csv`
- Batch source archive: `data/external/retailrocket/category_tree.csv`
- Staged Parquet data under `data/staged/source=retailrocket/`
- DuckDB warehouse at `data/warehouse/recomart.duckdb`
- Feature tables under `data/features/source=retailrocket/`
- Retailrocket model artifacts under `models/retailrocket/`
- Reports, plots, and assignment evidence under `reports/`

## Current Repository Structure

```text
configs/          Feature registry configuration
data/external/    DVC-tracked Retailrocket source archive
data/raw/         Raw Retailrocket ingestion snapshots
data/staged/      Cleaned Retailrocket Parquet datasets tracked by DVC
data/curated/     Curated analytical layer; populated in Phase 3
data/features/    Retailrocket ML feature datasets tracked by DVC
data/warehouse/   Generated DuckDB warehouse, ignored by Git
models/           Retailrocket model artifacts tracked by DVC
orchestration/    Prefect flows
reports/          CSV summaries, plots, markdown report, PDF report
sql/              Retailrocket DuckDB SQL scripts
src/              Python pipeline modules
```

## Environment

Use the existing conda environment:

```bash
conda activate recomart
```

If running commands without activating the environment, use:

```bash
conda run -n recomart <command>
```

## Retailrocket Raw Ingestion

Raw data uses this partition convention everywhere:

```text
data/raw/source=<source_name>/type=<data_type>/ingestion_timestamp=<YYYYMMDD_HHMMSS>/
```

Run Retailrocket batch ingestion from the external source archive:

```bash
python -m src.ingestion.ingest_retailrocket_batch
```

Batch ingestion lands:

```text
data/raw/source=retailrocket_batch/type=events/ingestion_timestamp=<YYYYMMDD_HHMMSS>/
data/raw/source=retailrocket_batch/type=item_properties_part1/ingestion_timestamp=<YYYYMMDD_HHMMSS>/
data/raw/source=retailrocket_batch/type=category_tree/ingestion_timestamp=<YYYYMMDD_HHMMSS>/
```

Run Retailrocket catalog API ingestion in mock/local mode:

```bash
RECOMART_CATALOG_API_MOCK_MODE=true python -m src.ingestion.ingest_retailrocket_catalog_api
```

API ingestion lands:

```text
data/raw/source=retailrocket_api/type=item_properties_delta/ingestion_timestamp=<YYYYMMDD_HHMMSS>/
```

Catalog API environment variables:

```text
RECOMART_CATALOG_API_BASE_URL
RECOMART_CATALOG_API_ENDPOINT
RECOMART_CATALOG_API_TIMEOUT_SEC
RECOMART_CATALOG_API_PAGE_SIZE
RECOMART_CATALOG_API_AUTH_TOKEN
RECOMART_CATALOG_API_STATE_FILE
RECOMART_CATALOG_API_MOCK_MODE
```

Default page size is 10 records per run. Cursor state is stored in `data/raw/source=retailrocket_api/_state/item_properties_delta_state.json` unless `RECOMART_CATALOG_API_STATE_FILE` is set.

The mock/demo API serves records from `data/external/retailrocket/item_properties_part2.csv` with a compatible endpoint:

```bash
uvicorn src.ingestion.mock_retailrocket_catalog_api:app --host 127.0.0.1 --port 8000
```

Then run the client against it:

```bash
RECOMART_CATALOG_API_BASE_URL=http://127.0.0.1:8000 python -m src.ingestion.ingest_retailrocket_catalog_api
```

## Retailrocket Commands

Inspect external Retailrocket files:

```bash
python -m src.ingestion.inspect_retailrocket_external
```

Prepare current staged Retailrocket datasets:

```bash
python -m src.preparation.prepare_retailrocket
```

Validate staged Retailrocket datasets:

```bash
python -m src.validation.validate_retailrocket_raw
python -m src.validation.validate_retailrocket_staged
```

Load the DuckDB warehouse:

```bash
python -m src.transformation.load_retailrocket_to_duckdb
```

Build Retailrocket feature tables:

```bash
python -m src.transformation.build_retailrocket_features
```

Train Retailrocket models:

```bash
python -m src.training.train_retailrocket_popularity_recommender
python -m src.training.train_retailrocket_content_recommender
```

Run Retailrocket inference:

```bash
python -m src.serving.recommend_retailrocket --model popularity --top-k 5
python -m src.serving.recommend_retailrocket --model content_based --top-k 5
python -m src.serving.recommend_retailrocket --model popularity --user-id 1327109 --top-k 10
python -m src.serving.recommend_retailrocket --model content_based --demo-users 1327109,925350,839657 --top-k 5
```

Run the full Retailrocket Prefect flow:

```bash
python -m orchestration.retailrocket_pipeline
```

Schedule only the Retailrocket catalog API ingestion every 30 minutes with Prefect 3:

```bash
prefect deploy orchestration/retailrocket_api_ingestion_flow.py:retailrocket_api_ingestion_flow \
  --name retailrocket-api-every-30-minutes \
  --interval 1800 \
  --pool default-agent-pool
```

For local execution without a real API server, the API ingestion flow defaults `RECOMART_CATALOG_API_MOCK_MODE=true` unless that environment variable is already set.

Regenerate assignment evidence:

```bash
python -m src.reporting.generate_assignment_evidence
python -m src.reporting.generate_model_comparison
python -m src.reporting.generate_final_report
python -m src.reporting.generate_assignment_pdf
```

## DVC

DVC tracks large Retailrocket data and model artifacts while Git stores source code, reports, and DVC metadata.

```bash
git status
dvc status
dvc pull
```

Tracked Retailrocket metadata currently includes:

- `data/external/retailrocket.dvc`
- `data/raw/source=retailrocket_batch.dvc`
- `data/raw/source=retailrocket_api.dvc`
- `data/staged/source=retailrocket.dvc`
- `data/curated/source=retailrocket.dvc`
- `data/features/source=retailrocket.dvc`
- `models/retailrocket.dvc`

## Current Refactor Status

This branch is moving the project to the final assignment story:

> RecoMart ingests Retailrocket interaction data from batch CSV files and item catalog metadata deltas from a REST API. It validates, stages, transforms, curates, versions, and uses this data to build Retailrocket recommendation features, a Retailrocket feature store, model training/evaluation, MLflow tracking, inference, and Prefect orchestration.

The following layers are completed or already present:

- Retailrocket external source archive
- Retailrocket batch raw ingestion
- Retailrocket REST/mock API raw ingestion
- Retailrocket raw validation
- Retailrocket staged data
- Retailrocket curated analytical datasets
- Retailrocket DuckDB warehouse
- Retailrocket feature tables
- Retailrocket popularity recommender
- Retailrocket content-based recommender
- Retailrocket inference script
- Retailrocket feature registry and retrieval demo
- Retailrocket-only Prefect orchestration flow
- Retailrocket 30-minute API ingestion flow documentation

The following layers are planned for the next phases:

- Retailrocket-only final report and PDF regeneration
- Clean rebuild workflow
