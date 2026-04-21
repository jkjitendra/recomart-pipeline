# RecoMart Recommendation Pipeline

## Data Management for Machine Learning – Assignment I

**Title:** End-to-End Data Management Pipeline for a Recommendation System

This project implements a complete, modular, and reproducible data management pipeline for a recommendation system for **RecoMart**, an e-commerce startup. The pipeline covers data ingestion, raw storage, validation, preparation, transformation, feature engineering, feature metadata management, data/model versioning, model training, experiment tracking, inference, orchestration, reporting, and PDF documentation.

---

## Table of Contents

- [1. Business Problem](#1-business-problem)
- [2. Datasets Used](#2-datasets-used)
  - [2.1 DummyJSON API Dataset](#21-dummyjson-api-dataset)
  - [2.2 Retailrocket Dataset](#22-retailrocket-dataset)
- [3. Tools and Technologies Used](#3-tools-and-technologies-used)
- [4. Repository Structure](#4-repository-structure)
- [5. Pipeline Stages](#5-pipeline-stages)
- [6. Data Ingestion](#6-data-ingestion)
- [7. Data Validation](#7-data-validation)
- [8. Data Preparation](#8-data-preparation)
- [9. DuckDB Warehouse and SQL Transformation](#9-duckdb-warehouse-and-sql-transformation)
- [10. Feature Engineering](#10-feature-engineering)
- [11. Feature Store / Feature Registry](#11-feature-store--feature-registry)
- [12. Data and Model Versioning with DVC](#12-data-and-model-versioning-with-dvc)
- [13. Model Training and Evaluation](#13-model-training-and-evaluation)
  - [13.1 Event-weighted Popularity Baseline](#131-event-weighted-popularity-baseline)
  - [13.2 Category/Content-based Recommender](#132-categorycontent-based-recommender)
  - [13.3 Model Comparison](#133-model-comparison)
- [14. MLflow Experiment Tracking](#14-mlflow-experiment-tracking)
- [15. Inference Demo](#15-inference-demo)
- [16. Pipeline Orchestration with Prefect](#16-pipeline-orchestration-with-prefect)
- [17. Reporting and Documentation](#17-reporting-and-documentation)
- [18. How to Run the Full Project](#18-how-to-run-the-full-project)
- [19. Key Deliverables](#19-key-deliverables)

---

## 1. Business Problem

RecoMart wants to improve customer engagement and cross-selling by recommending relevant products to users based on historical behavior and item/catalog metadata.

The pipeline is designed to support:

- Clean datasets for exploratory data analysis
- Feature tables for recommendation models
- Model training and evaluation
- Batch inference demonstrations
- Experiment tracking and reproducibility
- Automated end-to-end orchestration

---

## 2. Datasets Used

This project uses two data sources to satisfy the assignment requirement of ingesting multiple data types.

### 2.1 DummyJSON API Dataset

Used as a small API-based dataset to validate the full pipeline design.

Data types:

- Products
- Users
- Carts
- Cart items

Source type:

- REST API

Main scripts:

```bash
src/ingestion/fetch_dummyjson.py
src/ingestion/inspect_dummyjson_raw.py
src/preparation/prepare_dummyjson.py
```

### 2.2 Retailrocket Dataset

Used as the main large-scale recommendation dataset

Files:
```bash
data/external/retailrocket/events.csv
data/external/retailrocket/item_properties_part1.csv
data/external/retailrocket/item_properties_part2.csv
data/external/retailrocket/category_tree.csv
```

Data types:

Visitor-item interaction events
Item properties
Category hierarchy

Main scripts:
```bash
src/ingestion/inspect_retailrocket_external.py
src/preparation/prepare_retailrocket.py
src/validation/validate_retailrocket_staged.py
```

## 3. Tools and Technology Used

| Tool              | Purpose                                                                  |
| ----------------- | ------------------------------------------------------------------------ |
| Python            | Main implementation language                                             |
| pandas            | Data inspection, preparation, validation, reporting                      |
| DuckDB            | Local SQL warehouse and transformation engine                            |
| Parquet / PyArrow | Efficient staged and feature data storage                                |
| DVC               | Versioning external data, staged data, feature data, and model artifacts |
| Git               | Versioning source code, reports, and metadata                            |
| MLflow            | Experiment tracking, metrics, parameters, artifacts                      |
| Prefect           | End-to-end pipeline orchestration                                        |
| matplotlib        | EDA and model metric plots                                               |
| ReportLab         | Final PDF report generation                                              |
| Pickle            | Model artifact serialization                                             |


## 4. Repository Structure

```bash
configs/
  feature_store/
    dummyjson_feature_registry.json

data/
  external/
    retailrocket.dvc
  raw/
    source=dummyjson/
  staged/
    source=dummyjson.dvc
    source=retailrocket.dvc
  features/
    source=dummyjson.dvc
    source=retailrocket.dvc
  warehouse/
    recomart.duckdb

logs/
  ingestion_dummyjson.log

models/
  dummyjson.dvc
  retailrocket.dvc

orchestration/
  dummyjson_pipeline.py
  retailrocket_pipeline.py

reports/
  data_quality/
  plots/
  screenshots/
  final_project_report.md
  recomart_assignment_report.pdf
  reproducibility_commands.md

sql/
  dummyjson_warehouse.sql
  dummyjson_features.sql
  retailrocket_warehouse.sql
  retailrocket_features.sql

src/
  ingestion/
  preparation/
  validation/
  transformation/
  feature_store/
  training/
  serving/
  reporting/
```

## 5. Pipeline Stages

The implemented pipeline follows this flow:

```bash
Raw/API/External Data
  → Inspection
  → Validation
  → Staged Parquet
  → DuckDB Warehouse
  → Feature Engineering
  → Feature Metadata / Registry
  → Model Training
  → MLflow Experiment Tracking
  → Model Versioning with DVC
  → Inference Demo
  → Prefect Orchestration
  → Final Reporting
```

## 6. Data Ingestion

### 6.1 DummyJSON API ingestion

Fetches products, users, and carts from DummyJSON and stores them in a structured raw layout.

```bash
python -m src.ingestion.fetch_dummyjson
```

Raw layout example:
```bash
data/raw/source=dummyjson/type=products/ingest_date=YYYY-MM-DD/
data/raw/source=dummyjson/type=users/ingest_date=YYYY-MM-DD/
data/raw/source=dummyjson/type=carts/ingest_date=YYYY-MM-DD/
```

### 6.2 Retailrocket external inspection

Inspects the Retailrocket CSV files and generates summary reports.

```bash
python -m src.ingestion.inspect_retailrocket_external
```

Generated reports:
```bash
reports/retailrocket_external_summary.csv
reports/retailrocket_event_type_summary.csv
reports/retailrocket_top_item_properties.csv
reports/retailrocket_category_tree_summary.csv
```

## 7. Data Validation

Validation checks include:

 - File existence
 - Required columns
 - Row count checks
 - Missing values
 - Duplicate checks
 - Event type validity
 - Transaction ID consistency
 - Category metadata coverage
 - Availability metadata coverage

Run Retailrocket staged validation:
```bash
python -m src.validation.validate_retailrocket_staged
```

Run DummyJSON validation:
```bash
python -m src.validation.validate_dummyjson_raw
python -m src.validation.validate_dummyjson_staged
```

Validation reports:
```bash
reports/data_quality/dummyjson_raw_validation_report.csv
reports/data_quality/dummyjson_staged_validation_report.csv
reports/data_quality/retailrocket_staged_validation_report.csv
```

## 8. Data Preparation

### 8.1 DummyJSON preparation

Prepares DummyJSON data and stores it in staged Parquet format.

```bash
python -m src.preparation.prepare_dummyjson
```

Staged layout:
```bash
data/staged/source=dummyjson/type=products.parquet
data/staged/source=dummyjson/type=users.parquet
data/staged/source=dummyjson/type=carts.parquet
```

### 8.2 Retailrocket preparation

Prepares Retailrocket data and stores it in staged Parquet format.

```bash
python -m src.preparation.prepare_retailrocket
```

Staged layout:
```bash
data/staged/source=retailrocket/events.parquet
data/staged/source=retailrocket/item_properties.parquet
data/staged/source=retailrocket/category_tree.parquet
```

## 9. DuckDB Warehouse and SQL Transformation

DuckDB is used as the local analytical warehouse.

DummyJSON warehouse
```bash
python -m src.transformation.load_dummyjson_to_duckdb
```

Retailrocket warehouse
```bash
python -m src.transformation.load_retailrocket_to_duckdb
```

SQL files:
```bash
sql/dummyjson_warehouse.sql
sql/retailrocket_warehouse.sql
```

Warehouse path:
```bash
data/warehouse/recomart.duckdb
```

Example query:
```bash
duckdb data/warehouse/recomart.duckdb "SELECT event_type, COUNT(*) FROM mart.retailrocket_interactions GROUP BY event_type;"
```

## 10. Feature Engineering
DummyJSON features
```bash
python -m src.transformation.build_dummyjson_features
```

Retailrocket features
```bash
python -m src.transformation.build_retailrocket_features
```

Feature outputs:
```bash
data/features/source=dummyjson/
data/features/source=retailrocket/
```

Retailrocket feature groups:

 - User features
 - Item features
 - User-item interaction features

Generated feature reports:
```bash
reports/dummyjson_feature_summary.csv
reports/retailrocket_feature_summary.csv
reports/feature_logic_summary.csv
reports/feature_metadata_documentation.csv
```

## 11. Feature Store / Feature Registry

A custom feature registry is implemented for DummyJSON to demonstrate feature store concepts.

Registry file:
```bash
configs/feature_store/dummyjson_feature_registry.json
```

Feature retrieval demo:
```bash
python -m src.feature_store.retrieve_dummyjson_features
```

Generated reports:
```bash
reports/feature_store_registry_summary.csv
reports/feature_store_retrieval_demo.csv
```

## 12. Data and Model Versioning with DVC

DVC is used to track large datasets and model artifacts.

Tracked files include:
```bash
data/external/retailrocket.dvc
data/staged/source=dummyjson.dvc
data/staged/source=retailrocket.dvc
data/features/source=dummyjson.dvc
data/features/source=retailrocket.dvc
models/dummyjson.dvc
models/retailrocket.dvc
```

Check DVC status:
```bash
dvc status
```

If a DVC remote is configured, restore tracked artifacts with:
```bash
dvc pull
```

Versioning documentation:
```bash
reports/dvc_versioning_summary.csv
reports/repository_structure_summary.csv
```

## 13. Model Training and Evaluation

This project implements two Retailrocket recommendation models.

### 13.1 Event-weighted popularity baseline
```bash
python -m src.training.train_retailrocket_popularity_recommender
```

Model artifact:
```bash
models/retailrocket/popularity_recommender.pkl
```

Performance summary:
```bash
reports/retailrocket_model_training_summary.csv
```

### 13.2 Category/content-based recommender
```bash
python -m src.training.train_retailrocket_content_recommender
```

Model artifact:
```bash
models/retailrocket/content_based_recommender.pkl
```

Performance summary:
```bash
reports/retailrocket_content_model_training_summary.csv
```

The content-based recommender builds user profiles from historical item categories and parent categories, then ranks candidate items using:

 - Category similarity
 - Parent category similarity
 - Availability metadata
 - Conversion signals
 - Popularity prior

### 13.3 Model comparison
```bash
python -m src.reporting.generate_model_comparison
```

Generated report:
```bash
reports/retailrocket_model_comparison.csv
reports/plots/retailrocket_model_comparison_metrics.png
```

Current model comparison:
```bash
| Model                     | HitRate@10 | Precision@10 | Recall@10 |  NDCG@10 |
| ------------------------- | ---------: | -----------: | --------: | -------: |
| Popularity baseline       |   0.013432 |     0.001343 |  0.013432 | 0.008046 |
| Content-based recommender |   0.010691 |     0.001069 |  0.010691 | 0.004294 |
```

## 14. MLflow Experiment Tracking

MLflow is used to track:

 - Run IDs
 - Model parameters
 - Evaluation metrics
 - Model artifacts
 - Training reports
 - Candidate item reports

Experiment names:
```bash
dummyjson_recommendation_baselines
retailrocket_recommendation_models
```

### 14.1 Open MLflow UI:
Use the local SQLite tracking database:
```bash
mlflow ui --backend-store-uri "sqlite:///$PWD/mlflow.db" --port 5001
```

Then open:
```bash
http://127.0.0.1:5001
```

Expected experiments:
```bash
dummyjson_recommendation_baselines
retailrocket_recommendation_models
Default
```

### 14.2 Recreate MLflow Runs from Existing Artifacts

If the MLflow UI only shows the Default experiment, recreate the experiment records from the existing model artifacts and report files:
```bash
python -m src.reporting.recreate_mlflow_runs
```

Then open MLflow UI again:
```bash
mlflow ui --backend-store-uri "sqlite:///$PWD/mlflow.db" --port 5001
```

Then open:
```bash
http://127.0.0.1:5001
```

Expected experiments:
```bash
dummyjson_recommendation_baselines
retailrocket_recommendation_models
Default
```

This utility does not retrain models. It only logs existing summaries and artifacts back into MLflow.

### 14.3 MLflow Evidence Screenshots

MLflow evidence screenshots:
```bash
reports/screenshots/mlflow_01_experiments_list.png
reports/screenshots/mlflow_02_retailrocket_runs_table.png
reports/screenshots/mlflow_03_retailrocket_run_overview.png
reports/screenshots/mlflow_04_retailrocket_model_metrics.png
reports/screenshots/mlflow_05_dummyjson_runs_table.png
reports/screenshots/mlflow_06_retailrocket_content_based_run.png
```

Then run:

```bash
python -m py_compile src/reporting/recreate_mlflow_runs.py
git add README.md src/reporting/recreate_mlflow_runs.py
git commit -m "Document MLflow run recreation utility"
```

Then test the utility:
```bash
python -m src.reporting.recreate_mlflow_runs
mlflow ui --backend-store-uri "sqlite:///$PWD/mlflow.db" --port 5001
```

Open:
```bash
http://127.0.0.1:5001
```

## 15. Inference Demo

Run Retailrocket inference:
```bash
python -m src.serving.recommend_retailrocket
```

Run inference for one user:
```bash
python -m src.serving.recommend_retailrocket --user-id 1327109 --top-k 10
```

Run inference for custom demo users:
```bash
python -m src.serving.recommend_retailrocket --demo-users 1327109,925350,839657 --top-k 5
```

Generated inference reports:
```bash
reports/retailrocket_inference_demo_summary.csv
reports/retailrocket_inference_demo_recommendations.csv
```

## 16. Pipeline Orchestration with Prefect

The pipeline is orchestrated with Prefect.

DummyJSON pipeline
```bash
python -m orchestration.dummyjson_pipeline
```
Retailrocket pipeline
```bash
python -m orchestration.retailrocket_pipeline
```

Orchestration reports:
```bash
reports/orchestration_dummyjson_pipeline_summary.csv
reports/orchestration_retailrocket_pipeline_summary.csv
```

The Retailrocket orchestration flow runs:

```bash
inspect_retailrocket_external
prepare_retailrocket_staged
validate_retailrocket_staged
load_retailrocket_to_duckdb
build_retailrocket_features
train_retailrocket_popularity_recommender
run_retailrocket_inference
```

## 17. Reporting and Documentation

Generate assignment evidence artifacts:
```bash
python -m src.reporting.generate_assignment_evidence
```

Generate model comparison:
```bash
python -m src.reporting.generate_model_comparison
```

Generate Markdown report:
```bash
python -m src.reporting.generate_final_report
```

Generate final PDF report:
```bash
python -m src.reporting.generate_assignment_pdf
```

Important documentation outputs:
```bash
reports/final_project_report.md
reports/reproducibility_commands.md
reports/recomart_assignment_report.pdf
reports/plots/
reports/screenshots/
```

The main assignment PDF is:
```bash
reports/recomart_assignment_report.pdf
```

## 18. How to Run the Full Project
Step 1: Activate environment
```bash
conda activate recomart
```

Step 2: Verify Git and DVC state
```bash
git status
dvc status
```

Step 3: Restore DVC-tracked artifacts if required
```bash
dvc pull
```
If no DVC remote is configured, the DVC cache must already be available locally.

Step 4: Run DummyJSON pipeline
```bash
python -m orchestration.dummyjson_pipeline
```

Step 5: Run Retailrocket pipeline
```bash
python -m orchestration.retailrocket_pipeline
```

Step 6: Train content-based model
```bash
python -m src.training.train_retailrocket_content_recommender
```

Step 7: Generate comparison and final reports
```bash
python -m src.reporting.generate_assignment_evidence
python -m src.reporting.generate_model_comparison
python -m src.reporting.generate_final_report
python -m src.reporting.generate_assignment_pdf
```

Step 8: Open MLflow UI
```bash
TRACKING_URI=$(python -c 'import mlflow; print(mlflow.get_tracking_uri())')
mlflow ui --backend-store-uri "$TRACKING_URI" --port 5001
```

Open:
```bash
http://127.0.0.1:5001
```

19. Key Deliverables
```bash
| Deliverable                    | Location                                       |
| ------------------------------ | ---------------------------------------------- |
| Source code                    | `src/`, `sql/`, `orchestration/`               |
| Final PDF report               | `reports/recomart_assignment_report.pdf`      |
| Markdown report                | `reports/final_project_report.md`              |
| Reproducibility commands       | `reports/reproducibility_commands.md`          |
| Data quality reports           | `reports/data_quality/`                        |
| Feature metadata documentation | `reports/feature_metadata_documentation.csv`   |
| DVC versioning documentation   | `reports/dvc_versioning_summary.csv`           |
| SQL schema summary             | `reports/sql_schema_summary.csv`               |
| Summary plots                  | `reports/plots/`                               |
| MLflow screenshots             | `reports/screenshots/`                         |
| Orchestration reports          | `reports/orchestration_*_pipeline_summary.csv` |
| Model artifacts                | `models/*.dvc`                                 |
| DVC metadata                   | `data/**/*.dvc`, `models/*.dvc`                |
```
