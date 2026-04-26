# RecoMart Recommendation Data Pipeline

RecoMart is a reproducible data management pipeline for recommendation modeling on the Retailrocket dataset. The project currently focuses on Retailrocket interaction and catalog metadata, with the REST catalog-delta ingestion layer planned in the next refactor phase.

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
data/raw/         Raw ingestion layer; Retailrocket raw ingestion is added in Phase 2
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
python -m src.serving.recommend_retailrocket
python -m src.serving.recommend_retailrocket --user-id 1327109 --top-k 10
python -m src.serving.recommend_retailrocket --demo-users 1327109,925350,839657 --top-k 5
```

Run the current Retailrocket Prefect flow:

```bash
python -m orchestration.retailrocket_pipeline
```

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
- `data/staged/source=retailrocket.dvc`
- `data/features/source=retailrocket.dvc`
- `models/retailrocket.dvc`

## Current Refactor Status

This branch is moving the project to the final assignment story:

> RecoMart ingests Retailrocket interaction data from batch CSV files and item catalog metadata deltas from a REST API. It validates, stages, transforms, curates, versions, and uses this data to build Retailrocket recommendation features, a Retailrocket feature store, model training/evaluation, MLflow tracking, inference, and Prefect orchestration.

The following layers are completed or already present:

- Retailrocket external source archive
- Retailrocket staged data
- Retailrocket DuckDB warehouse
- Retailrocket feature tables
- Retailrocket popularity recommender
- Retailrocket content-based recommender
- Retailrocket inference script
- Retailrocket Prefect flow baseline

The following layers are planned for the next phases:

- Retailrocket batch raw ingestion
- Retailrocket REST catalog-delta ingestion
- Raw validation
- Curated analytical datasets
- Retailrocket feature registry and retrieval demo
- Retailrocket-only final report and PDF regeneration
- Clean rebuild workflow
