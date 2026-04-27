# RecoMart Video Walkthrough Script

Target length: 5-10 minutes.

## 1. Opening Introduction

Show:
- `README.md`
- `reports/recomart_assignment_report.pdf`

Script:
Introduce RecoMart as a Retailrocket-only recommendation data management pipeline. State that it covers ingestion, raw storage, validation, preparation, warehousing, feature engineering, feature store retrieval, model training, MLflow tracking, Prefect orchestration, inference, and final reporting.

## 2. Business Problem

Show:
- README section `Business Problem`
- PDF section `Problem Formulation`

Script:
Explain that the business goal is to recommend relevant products to users using interaction history and item metadata. Mention that the content-based recommender is the main assignment model and the popularity recommender is the benchmark.

## 3. Dataset and Architecture

Show:
- `data/external/retailrocket/`
- README section `Final Architecture`
- `reports/repository_structure_summary.csv`

Script:
Point out the Retailrocket batch CSV files: `events.csv`, `item_properties_part1.csv`, and `category_tree.csv`. Explain that `item_properties_part2.csv` is used by the REST/mock API as catalog metadata deltas.

## 4. Raw Ingestion: Batch + REST/Mock API

Show:
- `src/ingestion/ingest_retailrocket_batch.py`
- `src/ingestion/ingest_retailrocket_catalog_api.py`
- `src/ingestion/mock_retailrocket_catalog_api.py`
- `reports/retailrocket_batch_ingestion_summary.csv`
- `reports/retailrocket_api_ingestion_summary.csv`

Script:
Explain the raw partition layout:

```text
data/raw/source=<source_name>/type=<data_type>/ingestion_timestamp=<YYYYMMDD_HHMMSS>/
```

Mention that API ingestion defaults to 10 records per run, supports cursor state, timeout, retry, optional bearer token, and mock mode.

## 5. Data Layers

Show:
- `data/raw/source=retailrocket_batch.dvc`
- `data/raw/source=retailrocket_api.dvc`
- `data/staged/source=retailrocket.dvc`
- `data/curated/source=retailrocket.dvc`
- `data/features/source=retailrocket.dvc`
- `reports/retailrocket_preparation_summary.csv`
- `reports/retailrocket_curated_summary.csv`

Script:
Walk through external, raw, staged, curated, warehouse, feature, and model layers. Emphasize that preparation reads from raw landed files and that API delta preparation reads all available API partitions.

## 6. Validation and Data Quality

Show:
- `src/validation/validate_retailrocket_raw.py`
- `src/validation/validate_retailrocket_staged.py`
- `reports/data_quality/retailrocket_raw_validation_report.csv`
- `reports/data_quality/retailrocket_staged_validation_report.csv`

Script:
Explain schema checks, null checks, duplicate checks, event type checks, timestamp checks, latest metadata checks, and referential coverage between interactions and metadata.

## 7. Warehouse and Features

Show:
- `sql/retailrocket_warehouse.sql`
- `src/transformation/load_retailrocket_to_duckdb.py`
- `src/transformation/build_retailrocket_features.py`
- `reports/sql_schema_summary.csv`
- `reports/feature_logic_summary.csv`

Script:
Explain the DuckDB warehouse schemas: staged, curated, mart, and features. Show that user, item, and user-item feature tables are generated for training and inference.

## 8. Feature Store Demo

Show:
- `configs/feature_store/retailrocket_feature_registry.json`
- `src/feature_store/retrieve_retailrocket_features.py`
- `reports/feature_metadata_documentation.csv`
- `reports/feature_store_retrieval_demo.csv`

Script:
Explain that the custom feature registry documents feature views, entity keys, source paths, data types, feature roles, transformations, and usage. Show one user, one item, and one user-item retrieval sample.

## 9. DVC Versioning

Show:
- `reports/dvc_versioning_summary.csv`
- the `.dvc` files under `data/` and `models/`
- terminal command: `dvc status`

Script:
Explain that Git tracks source code, lightweight reports, and `.dvc` metadata while DVC tracks large generated data and model artifacts.

## 10. MLflow Model Tracking

Show:
- `reports/screenshots/mlflow_01_experiments_list.png`
- `reports/screenshots/mlflow_02_retailrocket_runs_table.png`
- `reports/screenshots/mlflow_03_retailrocket_run_overview.png`
- `reports/screenshots/mlflow_04_retailrocket_model_metrics.png`
- `reports/screenshots/mlflow_06_retailrocket_content_based_run.png`

Script:
Explain that both models log to the `retailrocket_recommendation_models` experiment with parameters, metrics, and artifacts.

## 11. Models and Metrics

Show:
- `src/training/train_retailrocket_popularity_recommender.py`
- `src/training/train_retailrocket_content_recommender.py`
- `reports/retailrocket_model_comparison.csv`
- `reports/plots/retailrocket_model_comparison_metrics.png`

Script:
Explain the popularity baseline and content-based recommender. Mention HitRate@10, Precision@10, Recall@10, and NDCG@10. State that content-based filtering uses item category, parent category, availability, conversion signals, and user category profiles.

## 12. Prefect Orchestration

Show:
- `orchestration/retailrocket_pipeline.py`
- `orchestration/retailrocket_api_ingestion_flow.py`
- `reports/orchestration_retailrocket_pipeline_summary.csv`

Script:
Show the 17-step Prefect pipeline summary and point out that all steps pass. Show the API ingestion deployment command:

```bash
prefect deploy orchestration/retailrocket_api_ingestion_flow.py:retailrocket_api_ingestion_flow \
  --name retailrocket-api-every-30-minutes \
  --interval 1800 \
  --pool default-agent-pool
```

## 13. Inference Demo

Show:
- `src/serving/recommend_retailrocket.py`
- `reports/retailrocket_inference_demo_summary.csv`
- `reports/retailrocket_inference_demo_recommendations.csv`

Script:
Show how to run:

```bash
python -m src.serving.recommend_retailrocket --model popularity --top-k 5
python -m src.serving.recommend_retailrocket --model content_based --top-k 5
```

Explain that previously seen items are filtered from recommendations.

## 14. Final Report and Submission Artifacts

Show:
- `reports/recomart_assignment_report.pdf`
- `reports/final_project_report.md`
- `reports/reproducibility_commands.md`
- `reports/submission_checklist.md`

Script:
Explain that the PDF is the single consolidated assignment report and maps the implementation to assignment tasks. Mention that the final ZIP is created only after final approval.

## 15. Closing Summary

Script:
Close by summarizing that RecoMart satisfies the assignment requirements using Retailrocket batch data, a REST/mock API metadata feed, DVC versioning, DuckDB transformations, custom feature store evidence, MLflow model tracking, Prefect orchestration, and reproducible documentation.
