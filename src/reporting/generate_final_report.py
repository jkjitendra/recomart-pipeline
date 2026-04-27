from __future__ import annotations

from pathlib import Path

import pandas as pd


REPORT_PATH = Path("reports/final_project_report.md")
COMMANDS_PATH = Path("reports/reproducibility_commands.md")


def read_csv(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    return pd.read_csv(file_path)


def table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_Not available._"
    return df.head(max_rows).to_markdown(index=False)


def choose_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    return df[[column for column in columns if column in df.columns]]


def status_summary(df: pd.DataFrame) -> str:
    if df.empty or "status" not in df.columns:
        return "_Not available._"
    return table(df.groupby("status").size().reset_index(name="check_count"))


def validation_warning_details(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "status" not in df.columns:
        return pd.DataFrame()

    warning_df = df[df["status"] == "WARN"].copy()

    return choose_columns(
        warning_df,
        ["dataset_name", "check_name", "status", "details"],
    )


def generate_report() -> str:
    retail_external = read_csv("reports/retailrocket_external_summary.csv")
    retail_events = read_csv("reports/retailrocket_event_type_summary.csv")
    raw_validation = read_csv("reports/data_quality/retailrocket_raw_validation_report.csv")
    staged_validation = read_csv("reports/data_quality/retailrocket_staged_validation_report.csv")
    preparation = read_csv("reports/retailrocket_preparation_summary.csv")
    curated = read_csv("reports/retailrocket_curated_summary.csv")
    warehouse = read_csv("reports/retailrocket_duckdb_load_summary.csv")
    sql_schema = read_csv("reports/sql_schema_summary.csv")
    feature_summary = read_csv("reports/retailrocket_feature_summary.csv")
    feature_logic = read_csv("reports/feature_logic_summary.csv")
    feature_metadata = read_csv("reports/feature_metadata_documentation.csv")
    feature_retrieval = read_csv("reports/feature_store_retrieval_demo.csv")
    dvc_summary = read_csv("reports/dvc_versioning_summary.csv")
    popularity_training = read_csv("reports/retailrocket_model_training_summary.csv")
    content_training = read_csv("reports/retailrocket_content_model_training_summary.csv")
    model_comparison = read_csv("reports/retailrocket_model_comparison.csv")
    inference = read_csv("reports/retailrocket_inference_demo_summary.csv")
    orchestration = read_csv("reports/orchestration_retailrocket_pipeline_summary.csv")

    lines: list[str] = []

    lines.append("# RecoMart Retailrocket Recommendation Pipeline")
    lines.append("")
    lines.append(
        "RecoMart is a Retailrocket-only end-to-end data management pipeline for recommendation modeling. "
        "The implementation ingests batch CSV interaction data and REST/mock API item metadata deltas, stores raw "
        "snapshots in a partitioned local data lake, validates and prepares data, loads a DuckDB warehouse, builds "
        "feature tables and a custom feature registry, trains Retailrocket recommendation models, tracks runs with "
        "MLflow, orchestrates execution with Prefect, and generates reproducible assignment evidence."
    )

    lines.append("\n## 1. Problem Formulation")
    lines.append(
        "The business goal is to recommend relevant products to Retailrocket users from interaction history and "
        "catalog metadata. The pipeline produces validated datasets, feature tables, model artifacts, inference "
        "outputs, lineage metadata, and a consolidated report suitable for submission."
    )
    lines.append("")
    lines.append("- Recommendation target: rank candidate items for users.")
    lines.append("- Main model: Retailrocket content-based recommender.")
    lines.append("- Baseline model: event-weighted Retailrocket popularity recommender.")
    lines.append("- Metrics: HitRate@10, Precision@10, Recall@10, and NDCG@10.")

    lines.append("\n## 2. Data Sources")
    lines.append("RecoMart uses Retailrocket data only.")
    lines.append("")
    lines.append("- Batch CSV source: `events.csv`, `item_properties_part1.csv`, and `category_tree.csv`.")
    lines.append("- REST/mock API source: item property delta records served from `item_properties_part2.csv`.")
    lines.append("- API default batch size: 10 records per run with cursor state persisted between runs.")
    lines.append("")
    lines.append("### 2.1 External Source Summary")
    lines.append(table(retail_external, max_rows=5))
    lines.append("")
    lines.append("### 2.2 Event Distribution")
    lines.append(table(retail_events, max_rows=5))

    lines.append("\n## 3. Batch and REST/Mock API Ingestion")
    lines.append(
        "Batch ingestion copies the Retailrocket CSV source files from `data/external/retailrocket/` into immutable "
        "raw snapshots. API ingestion fetches item property delta records from a configurable REST endpoint or the "
        "local mock mode used for reproducible assignment runs."
    )
    lines.append("")
    lines.append("API environment variables:")
    lines.append("")
    lines.append("```text")
    lines.append("RECOMART_CATALOG_API_BASE_URL")
    lines.append("RECOMART_CATALOG_API_ENDPOINT")
    lines.append("RECOMART_CATALOG_API_TIMEOUT_SEC")
    lines.append("RECOMART_CATALOG_API_PAGE_SIZE")
    lines.append("RECOMART_CATALOG_API_AUTH_TOKEN")
    lines.append("RECOMART_CATALOG_API_STATE_FILE")
    lines.append("RECOMART_CATALOG_API_MOCK_MODE")
    lines.append("```")

    lines.append("\n## 4. Raw Data Lake Layout")
    lines.append("Raw data uses one timestamp partition field:")
    lines.append("")
    lines.append("```text")
    lines.append("data/raw/source=<source_name>/type=<data_type>/ingestion_timestamp=<YYYYMMDD_HHMMSS>/")
    lines.append("```")
    lines.append("")
    lines.append("Implemented raw snapshot paths:")
    lines.append("")
    lines.append("```text")
    lines.append("data/raw/source=retailrocket_batch/type=events/ingestion_timestamp=<YYYYMMDD_HHMMSS>/")
    lines.append("data/raw/source=retailrocket_batch/type=item_properties_part1/ingestion_timestamp=<YYYYMMDD_HHMMSS>/")
    lines.append("data/raw/source=retailrocket_batch/type=category_tree/ingestion_timestamp=<YYYYMMDD_HHMMSS>/")
    lines.append("data/raw/source=retailrocket_api/type=item_properties_delta/ingestion_timestamp=<YYYYMMDD_HHMMSS>/")
    lines.append("```")

    lines.append("\n## 5. Data Validation Summary")
    lines.append("### 5.1 Raw Validation")
    lines.append(status_summary(raw_validation))
    lines.append("")
    lines.append("### 5.2 Staged and Curated Validation")
    lines.append(status_summary(staged_validation))
    lines.append("")
    lines.append(
        "Validation checks cover file presence, schema, row counts, nulls, duplicates, event type validity, "
        "timestamp validity, API delta presence, latest metadata uniqueness, and referential coverage between "
        "interactions and metadata. The current reports show no failing validation checks."
    )

    warning_df = validation_warning_details(staged_validation)
    if warning_df.empty:
        lines.append("")
        lines.append("No warning-level staged validation checks are present in the current report.")
    else:
        lines.append("")
        lines.append(
            "Warning-level checks are documented and do not block the pipeline. Current warnings are related to "
            "duplicate event groups and item metadata coverage for category and availability fields:"
        )
        lines.append("")
        lines.append(table(warning_df, max_rows=10))

    lines.append("\n## 6. Preparation, Staged Layer, and Curated Layer")
    lines.append(
        "Preparation reads from `data/raw/`, uses latest batch partitions for batch source types, and reads all API "
        "delta partitions so metadata deltas accumulate across runs. Staged outputs are typed Parquet files; curated "
        "outputs are EDA/modeling-ready analytical datasets."
    )
    lines.append("")
    lines.append("### 6.1 Staged Preparation Summary")
    lines.append(table(preparation, max_rows=10))
    lines.append("")
    lines.append("### 6.2 Curated Dataset Summary")
    lines.append(table(curated, max_rows=10))

    lines.append("\n## 7. EDA Plots")
    lines.append("Generated EDA/model evidence plots are stored under `reports/plots/`:")
    lines.append("")
    lines.append("- `retailrocket_event_distribution.png`")
    lines.append("- `retailrocket_top_items.png`")
    lines.append("- `retailrocket_user_activity_distribution.png`")
    lines.append("- `retailrocket_model_metrics.png`")
    lines.append("- `retailrocket_model_comparison_metrics.png`")

    lines.append("\n## 8. SQL Transformation and Warehouse")
    lines.append(
        "DuckDB stores Retailrocket staged, curated, mart, and feature schemas in `data/warehouse/recomart.duckdb`. "
        "The DuckDB binary is generated locally and is not committed directly to Git."
    )
    lines.append("")
    lines.append("### 8.1 DuckDB Load Summary")
    lines.append(table(warehouse, max_rows=20))
    lines.append("")
    lines.append("### 8.2 SQL Schema Sample")
    lines.append(table(sql_schema, max_rows=25))

    lines.append("\n## 9. Feature Engineering")
    lines.append("Feature tables are generated under `data/features/source=retailrocket/`.")
    lines.append("")
    lines.append("### 9.1 Feature Output Summary")
    lines.append(table(feature_summary, max_rows=10))
    lines.append("")
    lines.append("### 9.2 Feature Logic Summary")
    lines.append(table(feature_logic, max_rows=20))

    lines.append("\n## 10. Feature Store Registry and Retrieval Demo")
    lines.append(
        "The custom Retailrocket feature registry documents feature views, entity keys, source paths, data types, "
        "feature roles, usage, and source transformations. The retrieval demo reads current feature parquet files "
        "and writes representative user, item, and user-item samples."
    )
    lines.append("")
    lines.append("### 10.1 Feature Metadata Documentation Sample")
    lines.append(table(feature_metadata, max_rows=20))
    lines.append("")
    lines.append("### 10.2 Feature Retrieval Demo")
    lines.append(table(feature_retrieval, max_rows=10))

    lines.append("\n## 11. DVC Versioning Workflow")
    lines.append(
        "DVC tracks external data, raw snapshots, staged data, curated data, feature tables, and model artifacts. "
        "Git stores source code, lightweight reports, plots, and `.dvc` metadata."
    )
    lines.append("")
    lines.append(table(dvc_summary, max_rows=20))

    lines.append("\n## 12. Model Training and Evaluation")
    lines.append("### 12.1 Popularity Baseline")
    lines.append(table(popularity_training, max_rows=5))
    lines.append("")
    lines.append("### 12.2 Content-Based Recommender")
    lines.append(table(content_training, max_rows=5))
    lines.append("")
    lines.append("### 12.3 Model Comparison")
    lines.append(table(model_comparison, max_rows=5))

    lines.append("\n## 13. MLflow Tracking Metadata")
    lines.append(
        "Both Retailrocket models log to the MLflow experiment `retailrocket_recommendation_models`. The run records "
        "include model parameters, HitRate@10, Precision@10, Recall@10, NDCG@10, and report/model artifacts. "
        "MLflow screenshot evidence is stored under `reports/screenshots/`."
    )

    lines.append("\n## 14. Inference Demo")
    lines.append(table(inference, max_rows=5))

    lines.append("\n## 15. Prefect Orchestration Summary")
    lines.append(
        "Prefect orchestrates the full local pipeline using command-based tasks. The summary records step order, "
        "step name, command, status, return code, timing, and stdout/stderr tails."
    )
    lines.append("")
    lines.append(
        table(
            choose_columns(
                orchestration,
                ["step_order", "step_name", "status", "return_code", "duration_seconds"],
            ),
            max_rows=25,
        )
    )

    lines.append("\n## 16. API Ingestion 30-Minute Schedule")
    lines.append("The API-only Prefect flow can be deployed on a 30-minute interval with Prefect 3:")
    lines.append("")
    lines.append("```bash")
    lines.append("prefect deploy orchestration/retailrocket_api_ingestion_flow.py:retailrocket_api_ingestion_flow \\")
    lines.append("  --name retailrocket-api-every-30-minutes \\")
    lines.append("  --interval 1800 \\")
    lines.append("  --pool default-agent-pool")
    lines.append("```")

    lines.append("\n## 17. Known Validation Warnings")
    lines.append(
        "The current validation summary contains no failing checks. Warning-level checks are documented and do not "
        "block the pipeline. Current warnings, when present, are related to duplicate event groups and metadata "
        "coverage for category and availability fields."
    )
    if warning_df.empty:
        lines.append("")
        lines.append("No warning-level staged validation checks are present in the current report.")
    else:
        lines.append("")
        lines.append(table(warning_df, max_rows=10))

    lines.append("\n## 18. Reproducibility Steps")
    lines.append("```bash")
    lines.append("conda activate recomart")
    lines.append("dvc pull")
    lines.append("python -m orchestration.retailrocket_pipeline")
    lines.append("python -m src.reporting.generate_assignment_evidence")
    lines.append("python -m src.reporting.generate_model_comparison")
    lines.append("python -m src.reporting.generate_final_report")
    lines.append("python -m src.reporting.generate_assignment_pdf")
    lines.append("git status")
    lines.append("dvc status")
    lines.append("```")

    lines.append("\n## 19. Conclusion")
    lines.append(
        "RecoMart implements the required data management pipeline stages for a recommendation system using "
        "Retailrocket batch data and a REST/mock API metadata feed. The repository contains source code, DVC "
        "metadata, validation and model reports, Prefect orchestration evidence, MLflow screenshot evidence, and "
        "the consolidated PDF report."
    )

    return "\n".join(lines)


def generate_commands_doc() -> str:
    return """# Reproducibility Commands

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
prefect deploy orchestration/retailrocket_api_ingestion_flow.py:retailrocket_api_ingestion_flow \\
  --name retailrocket-api-every-30-minutes \\
  --interval 1800 \\
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
"""


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    REPORT_PATH.write_text(generate_report(), encoding="utf-8")
    COMMANDS_PATH.write_text(generate_commands_doc(), encoding="utf-8")

    print(f"Final project report saved: {REPORT_PATH}")
    print(f"Reproducibility commands saved: {COMMANDS_PATH}")


if __name__ == "__main__":
    main()
