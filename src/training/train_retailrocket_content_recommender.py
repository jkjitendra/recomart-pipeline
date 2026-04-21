from __future__ import annotations

import math
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import mlflow
import numpy as np
import pandas as pd


WAREHOUSE_PATH = Path("data/warehouse/recommart.duckdb")

MODEL_DIR = Path("models/retailrocket")
MODEL_PATH = MODEL_DIR / "content_based_recommender.pkl"

REPORTS_DIR = Path("reports")
TRAINING_SUMMARY_PATH = REPORTS_DIR / "retailrocket_content_model_training_summary.csv"
RECOMMENDATION_EXAMPLES_PATH = REPORTS_DIR / "retailrocket_content_recommendation_examples.csv"
CANDIDATE_ITEMS_PATH = REPORTS_DIR / "retailrocket_content_candidate_items.csv"

EXPERIMENT_NAME = "retailrocket_recommendation_models"
MODEL_NAME = "retailrocket_content_based_recommender"
MODEL_TYPE = "category_content_based_filtering"

TOP_K = 10
TEST_WINDOW_DAYS = 14
EVALUATION_SAMPLE_USERS = 10000
MODEL_CANDIDATE_ITEMS = 5000

EVENT_WEIGHTS = {
    "view": 1.0,
    "addtocart": 3.0,
    "transaction": 5.0,
}


def ensure_dirs() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def event_weight_expression() -> str:
    return """
        CASE event_type
            WHEN 'transaction' THEN 5.0
            WHEN 'addtocart' THEN 3.0
            ELSE 1.0
        END
    """


def get_timestamp_bounds(connection: duckdb.DuckDBPyConnection) -> tuple[int, int]:
    query = """
        SELECT
            MIN(event_timestamp_ms)::BIGINT AS min_timestamp_ms,
            MAX(event_timestamp_ms)::BIGINT AS max_timestamp_ms
        FROM mart.retailrocket_interactions;
    """

    row = connection.execute(query).fetchone()

    if row is None or row[0] is None or row[1] is None:
        raise ValueError("Could not calculate Retailrocket timestamp bounds.")

    return int(row[0]), int(row[1])


def get_candidate_items(
    connection: duckdb.DuckDBPyConnection,
    cutoff_timestamp_ms: int,
) -> pd.DataFrame:
    query = f"""
        WITH train_item_scores AS (
            SELECT
                item_id,
                COUNT(*) AS train_event_count,
                COUNT(DISTINCT user_id) AS train_unique_users,
                SUM(CASE WHEN event_type = 'view' THEN 1 ELSE 0 END) AS train_view_events,
                SUM(CASE WHEN event_type = 'addtocart' THEN 1 ELSE 0 END) AS train_addtocart_events,
                SUM(CASE WHEN event_type = 'transaction' THEN 1 ELSE 0 END) AS train_transaction_events,
                SUM({event_weight_expression()}) AS train_interaction_score
            FROM mart.retailrocket_interactions
            WHERE event_timestamp_ms <= {cutoff_timestamp_ms}
            GROUP BY item_id
        ),
        train_item_rates AS (
            SELECT
                *,
                COALESCE(
                    train_addtocart_events::DOUBLE / NULLIF(train_event_count, 0),
                    0.0
                ) AS addtocart_rate,
                COALESCE(
                    train_transaction_events::DOUBLE / NULLIF(train_event_count, 0),
                    0.0
                ) AS transaction_rate
            FROM train_item_scores
        )
        SELECT
            s.item_id,
            s.train_event_count,
            s.train_unique_users,
            s.train_view_events,
            s.train_addtocart_events,
            s.train_transaction_events,
            s.train_interaction_score,
            COALESCE(f.category_id, -1) AS category_id,
            COALESCE(f.parent_category_id, -1) AS parent_category_id,
            COALESCE(f.is_available, 0) AS is_available,
            COALESCE(f.has_category_metadata, 0) AS has_category_metadata,
            COALESCE(f.has_availability_metadata, 0) AS has_availability_metadata,
            s.addtocart_rate,
            s.transaction_rate,
            (
                LN(1 + s.train_interaction_score)
                + 0.50 * s.addtocart_rate
                + 1.00 * s.transaction_rate
                + 0.10 * COALESCE(f.is_available, 0)
            ) AS global_content_prior_score
        FROM train_item_rates s
        LEFT JOIN mart.retailrocket_item_features f
            ON s.item_id = f.item_id
        ORDER BY global_content_prior_score DESC, s.train_event_count DESC
        LIMIT {MODEL_CANDIDATE_ITEMS};
    """

    candidate_items_df = connection.execute(query).fetchdf()

    if candidate_items_df.empty:
        raise ValueError("No candidate items found for content-based recommender.")

    return candidate_items_df


def get_test_targets(
    connection: duckdb.DuckDBPyConnection,
    cutoff_timestamp_ms: int,
) -> pd.DataFrame:
    query = f"""
        WITH weighted_test AS (
            SELECT
                user_id,
                item_id AS target_item_id,
                MAX({event_weight_expression()}) AS target_weight,
                MIN(event_timestamp_ms) AS first_target_timestamp_ms
            FROM mart.retailrocket_interactions
            WHERE event_timestamp_ms > {cutoff_timestamp_ms}
              AND event_type IN ('addtocart', 'transaction')
            GROUP BY user_id, item_id
        ),
        ranked_targets AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY user_id
                    ORDER BY target_weight DESC, first_target_timestamp_ms ASC, target_item_id ASC
                ) AS target_rank
            FROM weighted_test
        )
        SELECT
            user_id,
            target_item_id,
            target_weight,
            first_target_timestamp_ms
        FROM ranked_targets
        WHERE target_rank = 1
        ORDER BY user_id
        LIMIT {EVALUATION_SAMPLE_USERS};
    """

    return connection.execute(query).fetchdf()


def get_train_seen_items(
    connection: duckdb.DuckDBPyConnection,
    cutoff_timestamp_ms: int,
    sampled_test_users_df: pd.DataFrame,
) -> dict[int, set[int]]:
    if sampled_test_users_df.empty:
        return {}

    user_ids = sampled_test_users_df["user_id"].astype(int).tolist()
    user_ids_sql = ",".join(str(user_id) for user_id in user_ids)

    query = f"""
        SELECT DISTINCT
            user_id,
            item_id
        FROM mart.retailrocket_interactions
        WHERE event_timestamp_ms <= {cutoff_timestamp_ms}
          AND user_id IN ({user_ids_sql});
    """

    seen_df = connection.execute(query).fetchdf()

    train_seen_items: dict[int, set[int]] = {}

    for row in seen_df.itertuples(index=False):
        user_id = int(row.user_id)
        item_id = int(row.item_id)
        train_seen_items.setdefault(user_id, set()).add(item_id)

    return train_seen_items


def get_user_content_profiles(
    connection: duckdb.DuckDBPyConnection,
    cutoff_timestamp_ms: int,
    sampled_test_users_df: pd.DataFrame,
) -> dict[int, dict[str, dict[int, float]]]:
    if sampled_test_users_df.empty:
        return {}

    user_ids = sampled_test_users_df["user_id"].astype(int).tolist()
    user_ids_sql = ",".join(str(user_id) for user_id in user_ids)

    query = f"""
        SELECT
            i.user_id,
            COALESCE(f.category_id, -1) AS category_id,
            COALESCE(f.parent_category_id, -1) AS parent_category_id,
            SUM({event_weight_expression()}) AS profile_weight
        FROM mart.retailrocket_interactions i
        LEFT JOIN mart.retailrocket_item_features f
            ON i.item_id = f.item_id
        WHERE i.event_timestamp_ms <= {cutoff_timestamp_ms}
          AND i.user_id IN ({user_ids_sql})
        GROUP BY
            i.user_id,
            COALESCE(f.category_id, -1),
            COALESCE(f.parent_category_id, -1);
    """

    profile_df = connection.execute(query).fetchdf()

    profiles: dict[int, dict[str, dict[int, float]]] = {}

    for row in profile_df.itertuples(index=False):
        user_id = int(row.user_id)
        category_id = int(row.category_id)
        parent_category_id = int(row.parent_category_id)
        weight = float(row.profile_weight)

        user_profile = profiles.setdefault(
            user_id,
            {
                "category_weights": {},
                "parent_category_weights": {},
            },
        )

        user_profile["category_weights"][category_id] = (
            user_profile["category_weights"].get(category_id, 0.0) + weight
        )
        user_profile["parent_category_weights"][parent_category_id] = (
            user_profile["parent_category_weights"].get(parent_category_id, 0.0) + weight
        )

    return profiles


def normalize_profile(profile: dict[str, dict[int, float]]) -> dict[str, dict[int, float]]:
    normalized = {
        "category_weights": {},
        "parent_category_weights": {},
    }

    for key in ["category_weights", "parent_category_weights"]:
        values = profile.get(key, {})
        total = sum(values.values())

        if total <= 0:
            normalized[key] = values
        else:
            normalized[key] = {
                category_id: weight / total
                for category_id, weight in values.items()
            }

    return normalized


def score_candidates_for_user(
    user_id: int,
    candidate_items_df: pd.DataFrame,
    user_profiles: dict[int, dict[str, dict[int, float]]],
    seen_items: set[int],
    top_k: int,
) -> list[dict[str, Any]]:
    raw_profile = user_profiles.get(
        user_id,
        {
            "category_weights": {},
            "parent_category_weights": {},
        },
    )
    profile = normalize_profile(raw_profile)

    category_weights = profile["category_weights"]
    parent_category_weights = profile["parent_category_weights"]

    scored_items: list[dict[str, Any]] = []

    for row in candidate_items_df.itertuples(index=False):
        item_id = int(row.item_id)

        if item_id in seen_items:
            continue

        category_id = int(row.category_id)
        parent_category_id = int(row.parent_category_id)

        category_score = category_weights.get(category_id, 0.0)
        parent_category_score = parent_category_weights.get(parent_category_id, 0.0)

        metadata_score = (
            0.10 * float(row.is_available)
            + 0.05 * float(row.has_category_metadata)
            + 0.05 * float(row.has_availability_metadata)
        )

        conversion_score = (
            0.50 * float(row.addtocart_rate)
            + 1.00 * float(row.transaction_rate)
        )

        popularity_prior = 0.05 * math.log1p(float(row.train_interaction_score))

        content_score = (
            2.00 * category_score
            + 1.00 * parent_category_score
            + metadata_score
            + conversion_score
            + popularity_prior
        )

        scored_items.append(
            {
                "user_id": user_id,
                "item_id": item_id,
                "content_score": content_score,
                "category_score": category_score,
                "parent_category_score": parent_category_score,
                "metadata_score": metadata_score,
                "conversion_score": conversion_score,
                "popularity_prior": popularity_prior,
                "category_id": category_id,
                "parent_category_id": parent_category_id,
                "is_available": int(row.is_available),
                "train_event_count": int(row.train_event_count),
                "train_interaction_score": float(row.train_interaction_score),
            }
        )

    scored_items.sort(
        key=lambda item: (
            item["content_score"],
            item["train_interaction_score"],
            item["train_event_count"],
        ),
        reverse=True,
    )

    top_items = scored_items[:top_k]

    for rank, item in enumerate(top_items, start=1):
        item["recommendation_rank"] = rank

    return top_items


def evaluate_model(
    test_targets_df: pd.DataFrame,
    candidate_items_df: pd.DataFrame,
    user_profiles: dict[int, dict[str, dict[int, float]]],
    train_seen_items: dict[int, set[int]],
    top_k: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if test_targets_df.empty:
        metrics = {
            "evaluated_users": 0,
            "hits": 0,
            f"hit_rate_at_{top_k}": 0.0,
            f"precision_at_{top_k}": 0.0,
            f"recall_at_{top_k}": 0.0,
            f"ndcg_at_{top_k}": 0.0,
        }
        return metrics, pd.DataFrame()

    hits = 0
    ndcg_sum = 0.0
    evaluated_users = 0
    example_rows: list[dict[str, Any]] = []

    for row in test_targets_df.itertuples(index=False):
        user_id = int(row.user_id)
        target_item_id = int(row.target_item_id)
        seen_items = train_seen_items.get(user_id, set())

        recommendations = score_candidates_for_user(
            user_id=user_id,
            candidate_items_df=candidate_items_df,
            user_profiles=user_profiles,
            seen_items=seen_items,
            top_k=top_k,
        )

        if not recommendations:
            continue

        evaluated_users += 1
        recommended_item_ids = [item["item_id"] for item in recommendations]

        if target_item_id in recommended_item_ids:
            hits += 1
            rank = recommended_item_ids.index(target_item_id) + 1
            ndcg_sum += 1.0 / math.log2(rank + 1)

        if len(example_rows) < 25:
            for item in recommendations[:5]:
                example_rows.append(
                    {
                        **item,
                        "target_item_id": target_item_id,
                        "is_hit": item["item_id"] == target_item_id,
                    }
                )

    hit_rate = hits / evaluated_users if evaluated_users else 0.0
    precision = hits / (evaluated_users * top_k) if evaluated_users else 0.0
    recall = hit_rate
    ndcg = ndcg_sum / evaluated_users if evaluated_users else 0.0

    metrics = {
        "evaluated_users": evaluated_users,
        "hits": hits,
        f"hit_rate_at_{top_k}": hit_rate,
        f"precision_at_{top_k}": precision,
        f"recall_at_{top_k}": recall,
        f"ndcg_at_{top_k}": ndcg,
    }

    examples_df = pd.DataFrame(example_rows)
    return metrics, examples_df


def save_model(
    candidate_items_df: pd.DataFrame,
    metrics: dict[str, Any],
    training_summary: dict[str, Any],
) -> None:
    model_payload = {
        "model_name": MODEL_NAME,
        "model_type": MODEL_TYPE,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "top_k": TOP_K,
        "test_window_days": TEST_WINDOW_DAYS,
        "candidate_item_count": len(candidate_items_df),
        "candidate_items": candidate_items_df.to_dict(orient="records"),
        "event_weights": EVENT_WEIGHTS,
        "metrics": metrics,
        "training_summary": training_summary,
        "scoring_logic": {
            "category_score_weight": 2.0,
            "parent_category_score_weight": 1.0,
            "metadata_score_weight": 1.0,
            "conversion_score_weight": 1.0,
            "popularity_prior_weight": 1.0,
        },
    }

    with MODEL_PATH.open("wb") as file:
        pickle.dump(model_payload, file)


def log_mlflow(training_summary: dict[str, Any]) -> str:
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=MODEL_NAME) as run:
        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_param("model_type", MODEL_TYPE)
        mlflow.log_param("top_k", TOP_K)
        mlflow.log_param("test_window_days", TEST_WINDOW_DAYS)
        mlflow.log_param("evaluation_sample_users", EVALUATION_SAMPLE_USERS)
        mlflow.log_param("model_candidate_items", MODEL_CANDIDATE_ITEMS)
        mlflow.log_param("event_weight_view", EVENT_WEIGHTS["view"])
        mlflow.log_param("event_weight_addtocart", EVENT_WEIGHTS["addtocart"])
        mlflow.log_param("event_weight_transaction", EVENT_WEIGHTS["transaction"])

        for key, value in training_summary.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                mlflow.log_metric(key, float(value))

        mlflow.log_artifact(str(MODEL_PATH))
        mlflow.log_artifact(str(TRAINING_SUMMARY_PATH))
        mlflow.log_artifact(str(RECOMMENDATION_EXAMPLES_PATH))
        mlflow.log_artifact(str(CANDIDATE_ITEMS_PATH))

        return run.info.run_id


def main() -> None:
    ensure_dirs()

    if not WAREHOUSE_PATH.exists():
        raise FileNotFoundError(
            f"Warehouse not found: {WAREHOUSE_PATH}. Run Retailrocket warehouse and feature steps first."
        )

    with duckdb.connect(str(WAREHOUSE_PATH)) as connection:
        min_timestamp_ms, max_timestamp_ms = get_timestamp_bounds(connection)
        cutoff_timestamp_ms = max_timestamp_ms - (TEST_WINDOW_DAYS * 24 * 60 * 60 * 1000)

        candidate_items_df = get_candidate_items(
            connection=connection,
            cutoff_timestamp_ms=cutoff_timestamp_ms,
        )

        test_targets_df = get_test_targets(
            connection=connection,
            cutoff_timestamp_ms=cutoff_timestamp_ms,
        )

        train_seen_items = get_train_seen_items(
            connection=connection,
            cutoff_timestamp_ms=cutoff_timestamp_ms,
            sampled_test_users_df=test_targets_df,
        )

        user_profiles = get_user_content_profiles(
            connection=connection,
            cutoff_timestamp_ms=cutoff_timestamp_ms,
            sampled_test_users_df=test_targets_df,
        )

    metrics, recommendation_examples_df = evaluate_model(
        test_targets_df=test_targets_df,
        candidate_items_df=candidate_items_df,
        user_profiles=user_profiles,
        train_seen_items=train_seen_items,
        top_k=TOP_K,
    )

    training_summary = {
        "model_name": MODEL_NAME,
        "model_type": MODEL_TYPE,
        "top_k": TOP_K,
        "test_window_days": TEST_WINDOW_DAYS,
        "evaluation_sample_users": EVALUATION_SAMPLE_USERS,
        "min_timestamp_ms": min_timestamp_ms,
        "max_timestamp_ms": max_timestamp_ms,
        "cutoff_timestamp_ms": cutoff_timestamp_ms,
        "candidate_item_count": len(candidate_items_df),
        "test_target_users": len(test_targets_df),
        "user_profile_count": len(user_profiles),
        **metrics,
    }

    save_model(
        candidate_items_df=candidate_items_df,
        metrics=metrics,
        training_summary=training_summary,
    )

    training_summary_df = pd.DataFrame([training_summary])
    training_summary_df.to_csv(TRAINING_SUMMARY_PATH, index=False)

    recommendation_examples_df.to_csv(RECOMMENDATION_EXAMPLES_PATH, index=False)
    candidate_items_df.to_csv(CANDIDATE_ITEMS_PATH, index=False)

    run_id = log_mlflow(training_summary)

    print("Retailrocket content-based recommender training completed.")
    print(f"Model saved: {MODEL_PATH}")
    print(f"Training summary saved: {TRAINING_SUMMARY_PATH}")
    print(f"Recommendation examples saved: {RECOMMENDATION_EXAMPLES_PATH}")
    print(f"Candidate items saved: {CANDIDATE_ITEMS_PATH}")
    print(f"MLflow experiment: {EXPERIMENT_NAME}")
    print(f"MLflow run_id: {run_id}")
    print()
    print(training_summary_df.to_string(index=False))


if __name__ == "__main__":
    main()