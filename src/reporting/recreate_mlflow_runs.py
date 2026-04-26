from __future__ import annotations

from pathlib import Path

import mlflow
import pandas as pd


TRACKING_DB = Path("mlflow.db").resolve()
TRACKING_URI = f"sqlite:///{TRACKING_DB}"

RETAIL_EXPERIMENT = "retailrocket_recommendation_models"


def log_metrics_from_row(row: pd.Series) -> None:
    for key, value in row.items():
        if isinstance(value, (int, float)) and pd.notna(value):
            mlflow.log_metric(key, float(value))


def log_artifacts(paths: list[str]) -> None:
    for path in paths:
        file_path = Path(path)
        if file_path.exists():
            mlflow.log_artifact(str(file_path))


def recreate_retailrocket_popularity_run() -> None:
    summary_path = Path("reports/retailrocket_model_training_summary.csv")
    if not summary_path.exists():
        print(f"Skipping Retailrocket popularity run. Missing: {summary_path}")
        return

    df = pd.read_csv(summary_path)
    row = df.iloc[0]

    mlflow.set_experiment(RETAIL_EXPERIMENT)

    with mlflow.start_run(run_name="retailrocket_popularity_recommender") as run:
        mlflow.log_param("model_name", "retailrocket_popularity_recommender")
        mlflow.log_param("model_type", "event_weighted_global_popularity")
        mlflow.log_param("dataset", "retailrocket")
        mlflow.log_param("top_k", int(row.get("top_k", 10)))
        mlflow.log_param("test_window_days", int(row.get("test_window_days", 14)))

        log_metrics_from_row(row)

        log_artifacts(
            [
                "models/retailrocket/popularity_recommender.pkl",
                "reports/retailrocket_model_training_summary.csv",
                "reports/retailrocket_recommendation_examples.csv",
                "reports/retailrocket_popularity_item_scores.csv",
            ]
        )

        print("Recreated Retailrocket popularity MLflow run:", run.info.run_id)


def recreate_retailrocket_content_run() -> None:
    summary_path = Path("reports/retailrocket_content_model_training_summary.csv")
    if not summary_path.exists():
        print(f"Skipping Retailrocket content-based run. Missing: {summary_path}")
        return

    df = pd.read_csv(summary_path)
    row = df.iloc[0]

    mlflow.set_experiment(RETAIL_EXPERIMENT)

    with mlflow.start_run(run_name="retailrocket_content_based_recommender") as run:
        mlflow.log_param("model_name", "retailrocket_content_based_recommender")
        mlflow.log_param("model_type", "category_content_based_filtering")
        mlflow.log_param("dataset", "retailrocket")
        mlflow.log_param("top_k", int(row.get("top_k", 10)))
        mlflow.log_param("test_window_days", int(row.get("test_window_days", 14)))

        log_metrics_from_row(row)

        log_artifacts(
            [
                "models/retailrocket/content_based_recommender.pkl",
                "reports/retailrocket_content_model_training_summary.csv",
                "reports/retailrocket_content_recommendation_examples.csv",
                "reports/retailrocket_content_candidate_items.csv",
            ]
        )

        print("Recreated Retailrocket content-based MLflow run:", run.info.run_id)


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)

    print("Using MLflow tracking URI:", mlflow.get_tracking_uri())

    recreate_retailrocket_popularity_run()
    recreate_retailrocket_content_run()

    print("\nExperiments after recreation:")
    for experiment in mlflow.search_experiments():
        print(experiment.experiment_id, experiment.name)


if __name__ == "__main__":
    main()
