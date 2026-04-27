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


def status_summary(df: pd.DataFrame) -> str:
    if df.empty or "status" not in df.columns:
        return "_Not available._"
    summary = df.groupby("status").size().reset_index(name="check_count")
    return table(summary)


def choose_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    return df[[column for column in columns if column in df.columns]]


def generate_report() -> str:
    retail_external = read_csv("reports/retailrocket_external_summary.csv")
    retail_events = read_csv("reports/retailrocket_event_type_summary.csv")
    retail_validation = read_csv("reports/data_quality/retailrocket_staged_validation_report.csv")
    retail_warehouse = read_csv("reports/retailrocket_duckdb_load_summary.csv")
    retail_features = read_csv("reports/retailrocket_feature_summary.csv")
    retail_training = read_csv("reports/retailrocket_model_training_summary.csv")
    retail_content_training = read_csv("reports/retailrocket_content_model_training_summary.csv")
    retail_model_comparison = read_csv("reports/retailrocket_model_comparison.csv")
    retail_inference = read_csv("reports/retailrocket_inference_demo_summary.csv")
    retail_orchestration = read_csv("reports/orchestration_retailrocket_pipeline_summary.csv")

    lines = []

    lines.append("# RecoMart: Retailrocket Recommendation Data Pipeline")
    lines.append("")
    lines.append(
        "RecoMart is a reproducible ML data pipeline for Retailrocket recommendation data. "
        "It covers data inspection, validation, staged storage, DuckDB warehousing, feature "
        "engineering, model training, MLflow tracking, DVC versioning, inference, and Prefect orchestration."
    )

    lines.append("\n# 1. Project Objective")
    lines.append(
        "The objective is to convert Retailrocket interaction and catalog metadata into validated "
        "features and trained recommender models for e-commerce recommendation use cases."
    )

    lines.append("\n# 2. Tools Used")
    lines.append(
        "| Tool | Usage |\n"
        "|---|---|\n"
        "| Python | Main implementation language |\n"
        "| pandas | Data inspection, summaries, reports |\n"
        "| DuckDB | Local analytical warehouse and SQL transformations |\n"
        "| Parquet | Staged and feature storage format |\n"
        "| DVC | Versioning large datasets and model artifacts |\n"
        "| Git | Versioning source code and lightweight reports |\n"
        "| MLflow | Experiment tracking and metric logging |\n"
        "| Prefect | Pipeline orchestration |\n"
    )

    lines.append("\n# 3. Repository Structure")
    lines.append(
        "```text\n"
        "configs/          Feature registry configuration\n"
        "data/external/    Retailrocket source archive tracked by DVC\n"
        "data/raw/         Raw ingestion snapshots; populated in the ingestion refactor\n"
        "data/staged/      Cleaned Retailrocket Parquet datasets tracked by DVC\n"
        "data/curated/     Curated analytical datasets; populated in the preparation refactor\n"
        "data/features/    Retailrocket ML feature datasets tracked by DVC\n"
        "data/warehouse/   Generated DuckDB local warehouse\n"
        "models/           Retailrocket model artifacts tracked by DVC\n"
        "reports/          CSV summaries, plots, final report, screenshots\n"
        "sql/              Retailrocket DuckDB SQL scripts\n"
        "src/              Python modules\n"
        "orchestration/    Prefect flows\n"
        "```"
    )

    lines.append("\n# 4. Retailrocket Dataset Summary")
    lines.append(table(retail_external))

    lines.append("\n## 4.1 Retailrocket Event Distribution")
    lines.append(table(retail_events))

    lines.append("\n# 5. Pipeline Architecture")
    lines.append(
        "```text\n"
        "External Retailrocket Source Archive\n"
        "  -> Raw Ingestion Layer\n"
        "  -> Validation\n"
        "  -> Staged Parquet\n"
        "  -> Curated Analytical Data\n"
        "  -> DuckDB Warehouse\n"
        "  -> Feature Tables\n"
        "  -> Model Training + MLflow\n"
        "  -> Model Artifacts + DVC\n"
        "  -> Inference Demo\n"
        "  -> Prefect Orchestration\n"
        "```"
    )

    lines.append("\n# 6. Data Quality Validation")
    lines.append(status_summary(retail_validation))
    lines.append(
        "Retailrocket validation checks include file existence, required columns, row counts, valid event types, "
        "transaction ID consistency, category metadata coverage, availability metadata coverage, and uniqueness "
        "of latest item metadata records."
    )

    lines.append("\n# 7. DuckDB Warehouse")
    lines.append(table(retail_warehouse))

    lines.append("\n# 8. Feature Engineering")
    lines.append(table(retail_features))
    lines.append(
        "Retailrocket features include user-level behavior, item-level popularity and metadata, and user-item "
        "interaction features such as event counts, implicit labels, and strongest event type."
    )

    lines.append("\n# 9. Feature Registry")
    lines.append(
        "A Retailrocket feature registry documents user, item, and user-item feature views. "
        "The retrieval demo reads current feature Parquet files under `data/features/source=retailrocket/` "
        "and writes sample retrieval evidence."
    )

    lines.append("\n# 10. Model Training")
    lines.append("\n## 10.1 Retailrocket Popularity Baseline")
    lines.append(table(retail_training))

    lines.append("\n## 10.2 Retailrocket Content-Based Recommender")
    lines.append(table(retail_content_training))
    lines.append(
        "The content-based recommender uses item category, parent category, availability metadata, "
        "conversion signals, and user category profiles to recommend items similar to a user's historical interests."
    )

    lines.append("\n## 10.3 Retailrocket Model Comparison")
    lines.append(table(retail_model_comparison))

    lines.append("\n# 11. Model Evaluation")
    if not retail_training.empty:
        row = retail_training.iloc[0]
        lines.append(
            f"The Retailrocket popularity baseline evaluated {int(row['evaluated_users'])} users and achieved "
            f"HitRate@10 = {row['hit_rate_at_10']:.6f}, "
            f"Precision@10 = {row['precision_at_10']:.6f}, "
            f"Recall@10 = {row['recall_at_10']:.6f}, "
            f"NDCG@10 = {row['ndcg_at_10']:.6f}."
        )

    if not retail_content_training.empty:
        row = retail_content_training.iloc[0]
        lines.append(
            f"The Retailrocket content-based recommender evaluated {int(row['evaluated_users'])} users and achieved "
            f"HitRate@10 = {row['hit_rate_at_10']:.6f}, "
            f"Precision@10 = {row['precision_at_10']:.6f}, "
            f"Recall@10 = {row['recall_at_10']:.6f}, "
            f"NDCG@10 = {row['ndcg_at_10']:.6f}."
        )

    lines.append("\n# 12. MLflow Experiment Tracking")
    lines.append(
        "MLflow logs Retailrocket model parameters, metrics, reports, and artifacts. Screenshots are stored "
        "in `reports/screenshots/`."
    )

    lines.append("\n# 13. DVC Versioning")
    lines.append(
        "DVC tracks Retailrocket external data, staged data, feature data, and model artifacts. "
        "This keeps Git lightweight while preserving reproducibility for large files."
    )

    lines.append("\n# 14. Inference Demo")
    lines.append(table(retail_inference))

    lines.append("\n# 15. Orchestration")
    lines.append(
        table(
            choose_columns(
                retail_orchestration,
                ["step_order", "step_name", "status", "duration_seconds"],
            )
        )
    )

    lines.append("\n# 16. Reproducibility")
    lines.append(
        "The project can be reproduced by activating the `recomart` environment, checking Git/DVC state, "
        "restoring artifacts if a DVC remote is configured, and running the Retailrocket pipeline commands."
    )

    lines.append("\n# 17. Limitations and Future Work")
    lines.append(
        "- The current models are lightweight recommenders suitable for the assignment scope\n"
        "- Future work can add collaborative filtering or sequence-aware recommendation models\n"
        "- A production deployment would use a managed API endpoint and persistent Prefect work pool\n"
        "- Phase 7 refreshes final narrative wording and PDF presentation for submission polish"
    )

    lines.append("\n# 18. Conclusion")
    lines.append(
        "The Retailrocket pipeline demonstrates a reproducible ML data management workflow with batch ingestion, "
        "REST catalog-delta ingestion, validation, preparation, warehousing, feature engineering, model training, "
        "MLflow tracking, inference, DVC versioning, and Prefect orchestration."
    )

    return "\n".join(lines)


def generate_commands_doc() -> str:
    lines = []

    lines.append("# Reproducibility Commands")
    lines.append("")
    lines.append("## Activate environment")
    lines.append("")
    lines.append("```bash")
    lines.append("conda activate recomart")
    lines.append("```")

    lines.append("")
    lines.append("## Check repository and DVC state")
    lines.append("")
    lines.append("```bash")
    lines.append("git status")
    lines.append("dvc status")
    lines.append("```")

    lines.append("")
    lines.append("## Restore DVC-tracked data and models if needed")
    lines.append("")
    lines.append("```bash")
    lines.append("dvc pull")
    lines.append("```")

    lines.append("")
    lines.append("## Run full Retailrocket Prefect pipeline")
    lines.append("")
    lines.append("```bash")
    lines.append("python -m orchestration.retailrocket_pipeline")
    lines.append("```")

    lines.append("")
    lines.append("## Schedule Retailrocket API ingestion every 30 minutes")
    lines.append("")
    lines.append("Prefect 3 local deployment command:")
    lines.append("")
    lines.append("```bash")
    lines.append("prefect deploy orchestration/retailrocket_api_ingestion_flow.py:retailrocket_api_ingestion_flow \\")
    lines.append("  --name retailrocket-api-every-30-minutes \\")
    lines.append("  --interval 1800 \\")
    lines.append("  --pool default-agent-pool")
    lines.append("```")
    lines.append("")
    lines.append("The API ingestion flow defaults to mock mode unless `RECOMART_CATALOG_API_MOCK_MODE` is already set.")

    lines.append("")
    lines.append("## Run Retailrocket training only")
    lines.append("")
    lines.append("```bash")
    lines.append("python -m src.training.train_retailrocket_popularity_recommender")
    lines.append("python -m src.training.train_retailrocket_content_recommender")
    lines.append("```")

    lines.append("")
    lines.append("## Run Retailrocket inference")
    lines.append("")
    lines.append("```bash")
    lines.append("python -m src.serving.recommend_retailrocket --model popularity --top-k 5")
    lines.append("python -m src.serving.recommend_retailrocket --model content_based --top-k 5")
    lines.append("python -m src.serving.recommend_retailrocket --model popularity --user-id 1327109 --top-k 10")
    lines.append("python -m src.serving.recommend_retailrocket --model content_based --demo-users 1327109,925350,839657 --top-k 5")
    lines.append("```")

    lines.append("")
    lines.append("## Check DuckDB warehouse")
    lines.append("")
    lines.append("```bash")
    lines.append('duckdb data/warehouse/recomart.duckdb "SELECT event_type, COUNT(*) FROM mart.retailrocket_interactions GROUP BY event_type;"')
    lines.append("```")

    lines.append("")
    lines.append("## Regenerate reports")
    lines.append("")
    lines.append("```bash")
    lines.append("python -m src.reporting.generate_assignment_evidence")
    lines.append("python -m src.reporting.generate_model_comparison")
    lines.append("python -m src.reporting.generate_final_report")
    lines.append("python -m src.reporting.generate_assignment_pdf")
    lines.append("```")

    return "\n".join(lines)


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    REPORT_PATH.write_text(generate_report(), encoding="utf-8")
    COMMANDS_PATH.write_text(generate_commands_doc(), encoding="utf-8")

    print(f"Final project report saved: {REPORT_PATH}")
    print(f"Reproducibility commands saved: {COMMANDS_PATH}")


if __name__ == "__main__":
    main()
