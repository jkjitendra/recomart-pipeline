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
    dummy_raw = read_csv("reports/dummyjson_raw_summary.csv")
    dummy_validation = read_csv("reports/data_quality/dummyjson_staged_validation_report.csv")
    dummy_features = read_csv("reports/dummyjson_feature_summary.csv")
    dummy_training = read_csv("reports/model_training_summary.csv")
    dummy_orchestration = read_csv("reports/orchestration_dummyjson_pipeline_summary.csv")

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

    lines.append("# Recommart: Data Management Pipeline for ML-Based Recommendation")
    lines.append("")
    lines.append(
        "This project implements an end-to-end data management and machine learning pipeline "
        "for recommendation systems. It covers ingestion, inspection, validation, staged storage, "
        "DuckDB warehousing, feature engineering, feature registry concepts, model training, "
        "MLflow tracking, DVC versioning, inference, and Prefect orchestration."
    )

    lines.append("\n# 1. Project Objective")
    lines.append(
        "The objective is to build a reproducible ML data pipeline that converts raw recommendation "
        "data into validated features and trained recommender models. DummyJSON is used as a small "
        "API-based demo dataset, while Retailrocket is used as the main large-scale recommendation dataset."
    )

    lines.append("\n# 2. Tools Used and Why")
    lines.append(
        "| Tool | Usage |\n"
        "|---|---|\n"
        "| Python | Main implementation language |\n"
        "| pandas | Data inspection, summaries, reports |\n"
        "| DuckDB | Local analytical warehouse and SQL transformation engine |\n"
        "| Parquet | Efficient staged and feature storage format |\n"
        "| DVC | Versioning large datasets and model artifacts |\n"
        "| Git | Versioning source code and lightweight reports |\n"
        "| MLflow | Experiment tracking and metric logging |\n"
        "| Prefect | Pipeline orchestration |\n"
        "| Docker | Available for containerized execution |\n"
    )

    lines.append("\n# 3. Repository Structure")
    lines.append(
        "```text\n"
        "configs/          Feature registry configuration\n"
        "data/external/    Retailrocket source data tracked by DVC\n"
        "data/raw/         DummyJSON API data\n"
        "data/staged/      Cleaned Parquet datasets tracked by DVC\n"
        "data/features/    ML feature datasets tracked by DVC\n"
        "data/warehouse/   DuckDB local warehouse\n"
        "models/           Trained model artifacts tracked by DVC\n"
        "reports/          CSV summaries, final report, screenshots\n"
        "sql/              DuckDB SQL scripts\n"
        "src/              Python modules\n"
        "orchestration/    Prefect flows\n"
        "```"
    )

    lines.append("\n# 4. Dataset Summary")
    lines.append("\n## 4.1 DummyJSON Raw Summary")
    lines.append(table(dummy_raw))

    lines.append("\n## 4.2 Retailrocket External Summary")
    lines.append(table(retail_external))

    lines.append("\n## 4.3 Retailrocket Event Distribution")
    lines.append(table(retail_events))

    lines.append("\n# 5. Pipeline Architecture")
    lines.append(
        "```text\n"
        "Raw/API/External Data\n"
        "  → Inspection\n"
        "  → Validation\n"
        "  → Staged Parquet\n"
        "  → DuckDB Warehouse\n"
        "  → Feature Tables\n"
        "  → Model Training + MLflow\n"
        "  → Model Artifacts + DVC\n"
        "  → Inference Demo\n"
        "  → Prefect Orchestration\n"
        "```"
    )

    lines.append("\n# 6. Data Quality Validation")
    lines.append("\n## 6.1 DummyJSON Validation Summary")
    lines.append(status_summary(dummy_validation))

    lines.append("\n## 6.2 Retailrocket Validation Summary")
    lines.append(status_summary(retail_validation))

    lines.append(
        "Retailrocket validation checks include file existence, required columns, row counts, valid event types, "
        "transaction ID consistency, category metadata coverage, availability metadata coverage, and uniqueness "
        "of latest item metadata records."
    )

    lines.append("\n# 7. DuckDB Warehouse")
    lines.append(
        "DuckDB is used as a local analytical warehouse. Staged Parquet files are loaded into structured tables "
        "and transformed into mart-level recommendation tables."
    )
    lines.append(table(retail_warehouse))

    lines.append("\n# 8. Feature Engineering")
    lines.append("\n## 8.1 DummyJSON Feature Summary")
    lines.append(table(dummy_features))

    lines.append("\n## 8.2 Retailrocket Feature Summary")
    lines.append(table(retail_features))

    lines.append(
        "Retailrocket features include user-level behavior, item-level popularity and metadata, and user-item "
        "interaction features such as event counts, implicit labels, and strongest event type."
    )

    lines.append("\n# 9. Feature Registry")
    lines.append(
        "A custom feature registry was implemented for DummyJSON to demonstrate feature store concepts. "
        "It maps entities, feature views, source Parquet paths, and intended use cases such as training, "
        "batch inference, and candidate filtering."
    )

    lines.append("\n# 10. Model Training")
    lines.append("\n## 10.1 DummyJSON Popularity Baseline")
    lines.append(table(dummy_training))

    lines.append("\n## 10.2 Retailrocket Popularity Baseline")
    lines.append(table(retail_training))

    lines.append("\n## 10.3 Retailrocket Content-Based Recommender")
    lines.append(table(retail_content_training))

    lines.append(
        "The content-based recommender uses item category, parent category, availability metadata, "
        "conversion signals, and user category profiles to recommend items similar to a user's historical "
        "interests. This directly satisfies the assignment requirement for a content-based recommendation model."
    )

    lines.append("\n## 10.4 Retailrocket Model Comparison")
    lines.append(table(retail_model_comparison))

    lines.append(
        "The Retailrocket modeling stage includes two models. The first is an event-weighted global "
        "popularity baseline. The second is a category/content-based recommender that builds user profiles "
        "from historical category interactions and ranks candidate items using category similarity, metadata, "
        "conversion rates, and popularity prior. Event weights are: `view = 1`, `addtocart = 3`, and `transaction = 5`."
    )

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
    lines.append(
        "NDCG@10 is included because it measures ranking quality. A hit at rank 1 receives more credit "
        "than a hit at rank 10."
    )

    lines.append("\n# 12. MLflow Experiment Tracking")
    lines.append(
        "MLflow was used to log model parameters, metrics, reports, and artifacts. Screenshots of the UI "
        "are stored in `reports/screenshots/`."
    )

    lines.append("\n# 13. DVC Versioning")
    lines.append(
        "DVC tracks external data, staged data, feature data, and model artifacts. This keeps Git lightweight "
        "while preserving reproducibility for large files."
    )

    lines.append("\n# 14. Inference Demo")
    lines.append(table(retail_inference))
    lines.append(
        "The Retailrocket inference script loads the trained model and generates recommendations for default "
        "active users, a single user, or custom comma-separated users. Previously interacted items are filtered."
    )

    lines.append("\n# 15. Orchestration")
    lines.append("\n## 15.1 DummyJSON Prefect Flow")
    lines.append(
        table(
            choose_columns(
                dummy_orchestration,
                ["step_order", "step_name", "status", "duration_seconds"],
            )
        )
    )

    lines.append("\n## 15.2 Retailrocket Prefect Flow")
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
        "The project can be reproduced by activating the environment, checking DVC state, restoring artifacts "
        "if a DVC remote is configured, and running the Prefect orchestration scripts."
    )

    lines.append("\n# 17. Team Work Division")
    lines.append(
        "| Team Member | Responsibility |\n"
        "|---|---|\n"
        "| Member 1 | Data ingestion and inspection |\n"
        "| Member 2 | Data preparation, validation, and DVC tracking |\n"
        "| Member 3 | DuckDB warehouse and feature engineering |\n"
        "| Member 4 | MLflow, training, inference, orchestration, and final reporting |\n"
    )

    lines.append("\n# 18. Limitations and Future Work")
    lines.append(
        "- Current Retailrocket models are lightweight popularity and content-based baselines\n"
        "- Add personalized collaborative filtering\n"
        "- Add matrix factorization or item-item recommendations\n"
        "- Add FastAPI serving endpoint\n"
        "- Add Streamlit monitoring dashboard\n"
        "- Add scheduled orchestration"
    )

    lines.append("\n# 19. Conclusion")
    lines.append(
        "The project successfully demonstrates a reproducible ML data management pipeline for recommendation "
        "systems using modern tools such as DuckDB, DVC, MLflow, and Prefect."
    )

    return "\n".join(lines)


def generate_commands_doc() -> str:
    lines = []

    lines.append("# Reproducibility Commands")
    lines.append("")
    lines.append("## Activate environment")
    lines.append("")
    lines.append("```bash")
    lines.append("conda activate recommart")
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
    lines.append("If no DVC remote is configured, the DVC cache must already exist locally.")

    lines.append("")
    lines.append("## Run DummyJSON pipeline")
    lines.append("")
    lines.append("```bash")
    lines.append("python -m orchestration.dummyjson_pipeline")
    lines.append("```")

    lines.append("")
    lines.append("## Run Retailrocket pipeline")
    lines.append("")
    lines.append("```bash")
    lines.append("python -m orchestration.retailrocket_pipeline")
    lines.append("```")

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
    lines.append("python -m src.serving.recommend_retailrocket")
    lines.append("python -m src.serving.recommend_retailrocket --user-id 1327109 --top-k 10")
    lines.append("python -m src.serving.recommend_retailrocket --demo-users 1327109,925350,839657 --top-k 5")
    lines.append("```")

    lines.append("")
    lines.append("## Open MLflow UI")
    lines.append("")
    lines.append("Use the same MLflow tracking URI used by the training scripts.")
    lines.append("")
    lines.append("```bash")
    lines.append("TRACKING_URI=$(python -c 'import mlflow; print(mlflow.get_tracking_uri())')")
    lines.append('mlflow ui --backend-store-uri "$TRACKING_URI" --port 5001')
    lines.append("```")
    lines.append("")
    lines.append("Open `http://127.0.0.1:5001`, then go to `model training → Experiments`.")

    lines.append("")
    lines.append("## Check DuckDB warehouse")
    lines.append("")
    lines.append("```bash")
    lines.append('duckdb data/warehouse/recommart.duckdb "SELECT event_type, COUNT(*) FROM mart.retailrocket_interactions GROUP BY event_type;"')
    lines.append("```")

    lines.append("")
    lines.append("## Regenerate final report")
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