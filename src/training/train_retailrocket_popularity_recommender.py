import pickle
from pathlib import Path
from typing import Any

import duckdb
import mlflow
import pandas as pd
import math


WAREHOUSE_PATH = Path("data/warehouse/recommart.duckdb")

MODEL_DIR = Path("models/retailrocket")
MODEL_PATH = MODEL_DIR / "popularity_recommender.pkl"

TRAINING_SUMMARY_PATH = Path("reports/retailrocket_model_training_summary.csv")
RECOMMENDATION_EXAMPLES_PATH = Path("reports/retailrocket_recommendation_examples.csv")
POPULARITY_SCORES_PATH = Path("reports/retailrocket_popularity_item_scores.csv")

EXPERIMENT_NAME = "retailrocket_recommendation_models"
MODEL_NAME = "retailrocket_popularity_recommender"

TOP_K = 10
TEST_WINDOW_DAYS = 14
EVALUATION_SAMPLE_USERS = 10000
MODEL_TOP_N_ITEMS = 5000


def ensure_prerequisites() -> None:
    if not WAREHOUSE_PATH.exists():
        raise FileNotFoundError(
            f"Warehouse not found: {WAREHOUSE_PATH}. "
            "Run python -m src.transformation.load_retailrocket_to_duckdb first."
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_cutoff_timestamp_ms(connection: duckdb.DuckDBPyConnection) -> tuple[int, int, str, str]:
    result = connection.execute(
        """
        SELECT
            MIN(event_timestamp_ms) AS min_timestamp_ms,
            MAX(event_timestamp_ms) AS max_timestamp_ms,
            MIN(event_datetime) AS min_event_datetime,
            MAX(event_datetime) AS max_event_datetime
        FROM staged.retailrocket_events
        """
    ).fetchdf().iloc[0]

    min_timestamp_ms = int(result["min_timestamp_ms"])
    max_timestamp_ms = int(result["max_timestamp_ms"])
    min_datetime = str(result["min_event_datetime"])
    max_datetime = str(result["max_event_datetime"])

    cutoff_timestamp_ms = max_timestamp_ms - (TEST_WINDOW_DAYS * 24 * 60 * 60 * 1000)

    return min_timestamp_ms, max_timestamp_ms, min_datetime, max_datetime, cutoff_timestamp_ms


def compute_item_scores(
    connection: duckdb.DuckDBPyConnection,
    cutoff_timestamp_ms: int,
) -> pd.DataFrame:
    """
    Computes event-weighted item popularity from the training period.

    The popularity score gives more importance to:
    - total weighted interaction score
    - number of unique users
    - transaction events
    - add-to-cart events
    - item availability
    """
    query = f"""
    WITH train_events AS (
        SELECT
            visitor_id AS user_id,
            item_id,
            event_type,
            CASE
                WHEN event_type = 'view' THEN 1.0
                WHEN event_type = 'addtocart' THEN 3.0
                WHEN event_type = 'transaction' THEN 5.0
                ELSE 0.0
            END AS event_weight
        FROM staged.retailrocket_events
        WHERE event_timestamp_ms < {cutoff_timestamp_ms}
    ),
    item_agg AS (
        SELECT
            item_id,
            COUNT(*) AS train_event_count,
            COUNT(DISTINCT user_id) AS train_unique_users,
            SUM(CASE WHEN event_type = 'view' THEN 1 ELSE 0 END) AS train_view_events,
            SUM(CASE WHEN event_type = 'addtocart' THEN 1 ELSE 0 END) AS train_addtocart_events,
            SUM(CASE WHEN event_type = 'transaction' THEN 1 ELSE 0 END) AS train_transaction_events,
            SUM(event_weight) AS train_interaction_score
        FROM train_events
        GROUP BY item_id
    )
    SELECT
        agg.item_id,
        agg.train_event_count,
        agg.train_unique_users,
        agg.train_view_events,
        agg.train_addtocart_events,
        agg.train_transaction_events,
        agg.train_interaction_score,
        item.category_id,
        item.parent_category_id,
        item.is_available,
        item.has_category_metadata,
        item.has_availability_metadata,
        CASE
            WHEN agg.train_view_events > 0
                THEN CAST(agg.train_addtocart_events AS DOUBLE) / agg.train_view_events
            ELSE 0.0
        END AS addtocart_rate,
        CASE
            WHEN agg.train_view_events > 0
                THEN CAST(agg.train_transaction_events AS DOUBLE) / agg.train_view_events
            ELSE 0.0
        END AS transaction_rate,
        (
            0.40 * LN(1 + agg.train_interaction_score)
            + 0.25 * LN(1 + agg.train_unique_users)
            + 0.20 * LN(1 + agg.train_transaction_events)
            + 0.10 * LN(1 + agg.train_addtocart_events)
            + 0.05 * COALESCE(item.is_available, 0)
        ) AS popularity_score
    FROM item_agg agg
    LEFT JOIN features.retailrocket_item_features item
        ON agg.item_id = item.item_id
    ORDER BY popularity_score DESC, train_interaction_score DESC, train_unique_users DESC
    """

    item_scores_df = connection.execute(query).fetchdf()
    item_scores_df["rank"] = range(1, len(item_scores_df) + 1)

    return item_scores_df


def build_test_targets(
    connection: duckdb.DuckDBPyConnection,
    cutoff_timestamp_ms: int,
    evaluation_sample_users: int,
) -> pd.DataFrame:
    """
    Builds one future target item per user from the test window.

    We focus on high-intent future actions:
    - addtocart
    - transaction

    For each user, we choose the strongest/latest test item.
    """
    query = f"""
    WITH test_events AS (
        SELECT
            visitor_id AS user_id,
            item_id,
            event_type,
            event_timestamp_ms,
            CASE
                WHEN event_type = 'transaction' THEN 5.0
                WHEN event_type = 'addtocart' THEN 3.0
                ELSE 1.0
            END AS event_weight
        FROM staged.retailrocket_events
        WHERE event_timestamp_ms >= {cutoff_timestamp_ms}
          AND event_type IN ('addtocart', 'transaction')
    ),
    test_item_agg AS (
        SELECT
            user_id,
            item_id,
            MAX(event_weight) AS target_weight,
            MAX(event_timestamp_ms) AS last_test_timestamp_ms
        FROM test_events
        GROUP BY user_id, item_id
    ),
    ranked_targets AS (
        SELECT
            user_id,
            item_id,
            target_weight,
            last_test_timestamp_ms,
            ROW_NUMBER() OVER (
                PARTITION BY user_id
                ORDER BY target_weight DESC, last_test_timestamp_ms DESC, item_id
            ) AS row_number
        FROM test_item_agg
    )
    SELECT
        user_id,
        item_id AS target_item_id,
        target_weight,
        last_test_timestamp_ms
    FROM ranked_targets
    WHERE row_number = 1
    ORDER BY hash(user_id)
    LIMIT {evaluation_sample_users}
    """

    return connection.execute(query).fetchdf()


def get_train_seen_items(
    connection: duckdb.DuckDBPyConnection,
    cutoff_timestamp_ms: int,
    sampled_test_users_df: pd.DataFrame,
) -> dict[int, set[int]]:
    """
    Gets training-period item history for sampled test users.

    This is used to avoid recommending items already seen in training.
    """
    if sampled_test_users_df.empty:
        return {}

    connection.register(
        "sampled_test_users_df",
        sampled_test_users_df[["user_id"]],
    )

    query = f"""
    SELECT DISTINCT
        events.visitor_id AS user_id,
        events.item_id
    FROM staged.retailrocket_events events
    INNER JOIN sampled_test_users_df users
        ON events.visitor_id = users.user_id
    WHERE events.event_timestamp_ms < {cutoff_timestamp_ms}
    """

    seen_df = connection.execute(query).fetchdf()

    history: dict[int, set[int]] = {}

    for row in seen_df.itertuples(index=False):
        user_id = int(row.user_id)
        item_id = int(row.item_id)

        if user_id not in history:
            history[user_id] = set()

        history[user_id].add(item_id)

    return history


def recommend_for_user(
    item_ids_ranked: list[int],
    seen_items: set[int],
    top_k: int,
) -> list[int]:
    recommendations = []

    for item_id in item_ids_ranked:
        if item_id in seen_items:
            continue

        recommendations.append(item_id)

        if len(recommendations) >= top_k:
            break

    return recommendations


def evaluate_model(
    test_targets_df: pd.DataFrame,
    train_seen_items: dict[int, set[int]],
    item_scores_df: pd.DataFrame,
    top_k: int,
) -> dict[str, Any]:
    """
    Evaluate top-K recommendations.

    Metrics:
    - HitRate@K: fraction of evaluated users where the target item appears in top-K.
    - Precision@K: hits divided by total recommended slots.
    - Recall@K: same as HitRate here because each user has one target item.
    - NDCG@K: ranking-aware score. A hit at rank 1 is better than a hit at rank 10.
    """
    if test_targets_df.empty:
        return {
            "evaluated_users": 0,
            "hits": 0,
            f"hit_rate_at_{top_k}": 0.0,
            f"precision_at_{top_k}": 0.0,
            f"recall_at_{top_k}": 0.0,
            f"ndcg_at_{top_k}": 0.0,
        }

    item_ids_ranked = item_scores_df["item_id"].astype(int).tolist()

    hits = 0
    ndcg_sum = 0.0
    evaluated_users = 0

    for row in test_targets_df.itertuples(index=False):
        user_id = int(row.user_id)
        target_item_id = int(row.target_item_id)

        seen_items = train_seen_items.get(user_id, set())

        recommendations = [
            item_id
            for item_id in item_ids_ranked
            if item_id not in seen_items
        ][:top_k]

        if not recommendations:
            continue

        evaluated_users += 1

        if target_item_id in recommendations:
            hits += 1

            rank = recommendations.index(target_item_id) + 1
            ndcg_sum += 1.0 / math.log2(rank + 1)

    hit_rate = hits / evaluated_users if evaluated_users else 0.0
    precision = hits / (evaluated_users * top_k) if evaluated_users else 0.0
    recall = hit_rate
    ndcg = ndcg_sum / evaluated_users if evaluated_users else 0.0

    return {
        "evaluated_users": evaluated_users,
        "hits": hits,
        f"hit_rate_at_{top_k}": hit_rate,
        f"precision_at_{top_k}": precision,
        f"recall_at_{top_k}": recall,
        f"ndcg_at_{top_k}": ndcg,
    }


def build_recommendation_examples(
    test_targets_df: pd.DataFrame,
    train_seen_items: dict[int, set[int]],
    item_scores_df: pd.DataFrame,
    top_k: int = 5,
    user_count: int = 5,
) -> pd.DataFrame:
    item_ids_ranked = item_scores_df["item_id"].astype(int).tolist()
    item_lookup = item_scores_df.set_index("item_id").to_dict(orient="index")

    rows = []

    example_users = (
        test_targets_df["user_id"]
        .drop_duplicates()
        .astype(int)
        .head(user_count)
        .tolist()
    )

    for user_id in example_users:
        seen_items = train_seen_items.get(user_id, set())

        recommendations = recommend_for_user(
            item_ids_ranked=item_ids_ranked,
            seen_items=seen_items,
            top_k=top_k,
        )

        target_items = test_targets_df.loc[
            test_targets_df["user_id"] == user_id,
            "target_item_id",
        ].astype(int).tolist()

        target_item_id = target_items[0] if target_items else None

        for rank, item_id in enumerate(recommendations, start=1):
            metadata = item_lookup.get(item_id, {})

            rows.append(
                {
                    "user_id": user_id,
                    "recommendation_rank": rank,
                    "item_id": item_id,
                    "target_item_id": target_item_id,
                    "is_hit": item_id == target_item_id,
                    "popularity_score": metadata.get("popularity_score"),
                    "train_event_count": metadata.get("train_event_count"),
                    "train_unique_users": metadata.get("train_unique_users"),
                    "train_addtocart_events": metadata.get("train_addtocart_events"),
                    "train_transaction_events": metadata.get("train_transaction_events"),
                    "category_id": metadata.get("category_id"),
                    "parent_category_id": metadata.get("parent_category_id"),
                    "is_available": metadata.get("is_available"),
                }
            )

    return pd.DataFrame(rows)


def save_model_artifact(
    item_scores_df: pd.DataFrame,
    training_summary: dict[str, Any],
) -> None:
    top_item_scores_df = item_scores_df.head(MODEL_TOP_N_ITEMS).copy()

    model_artifact = {
        "model_name": MODEL_NAME,
        "model_type": "event_weighted_global_popularity",
        "top_k": TOP_K,
        "model_top_n_items": MODEL_TOP_N_ITEMS,
        "top_items": top_item_scores_df["item_id"].astype(int).tolist(),
        "item_scores": top_item_scores_df.to_dict(orient="records"),
        "metrics": training_summary,
    }

    with MODEL_PATH.open("wb") as file:
        pickle.dump(model_artifact, file)


def log_mlflow(training_summary: dict[str, Any]) -> str:
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=MODEL_NAME) as run:
        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_param("model_type", "event_weighted_global_popularity")
        mlflow.log_param("top_k", TOP_K)
        mlflow.log_param("test_window_days", TEST_WINDOW_DAYS)
        mlflow.log_param("evaluation_sample_users", EVALUATION_SAMPLE_USERS)
        mlflow.log_param("model_top_n_items", MODEL_TOP_N_ITEMS)

        for key, value in training_summary.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, float(value))

        mlflow.log_artifact(str(MODEL_PATH))
        mlflow.log_artifact(str(TRAINING_SUMMARY_PATH))
        mlflow.log_artifact(str(RECOMMENDATION_EXAMPLES_PATH))
        mlflow.log_artifact(str(POPULARITY_SCORES_PATH))

        return run.info.run_id


def main() -> None:
    ensure_prerequisites()

    with duckdb.connect(str(WAREHOUSE_PATH)) as connection:
        (
            min_timestamp_ms,
            max_timestamp_ms,
            min_datetime,
            max_datetime,
            cutoff_timestamp_ms,
        ) = get_cutoff_timestamp_ms(connection)

        train_event_count = int(
            connection.execute(
                f"""
                SELECT COUNT(*)
                FROM staged.retailrocket_events
                WHERE event_timestamp_ms < {cutoff_timestamp_ms}
                """
            ).fetchone()[0]
        )

        test_event_count = int(
            connection.execute(
                f"""
                SELECT COUNT(*)
                FROM staged.retailrocket_events
                WHERE event_timestamp_ms >= {cutoff_timestamp_ms}
                """
            ).fetchone()[0]
        )

        item_scores_df = compute_item_scores(
            connection=connection,
            cutoff_timestamp_ms=cutoff_timestamp_ms,
        )

        test_targets_df = build_test_targets(
            connection=connection,
            cutoff_timestamp_ms=cutoff_timestamp_ms,
            evaluation_sample_users=EVALUATION_SAMPLE_USERS,
        )

        train_seen_items = get_train_seen_items(
            connection=connection,
            cutoff_timestamp_ms=cutoff_timestamp_ms,
            sampled_test_users_df=test_targets_df,
        )

    metrics = evaluate_model(
        test_targets_df=test_targets_df,
        train_seen_items=train_seen_items,
        item_scores_df=item_scores_df,
        top_k=TOP_K,
    )

    recommendation_examples_df = build_recommendation_examples(
        test_targets_df=test_targets_df,
        train_seen_items=train_seen_items,
        item_scores_df=item_scores_df,
        top_k=5,
        user_count=5,
    )

    train_items = int(item_scores_df["item_id"].nunique())
    saved_model_items = int(min(MODEL_TOP_N_ITEMS, len(item_scores_df)))
    model_item_coverage = saved_model_items / train_items if train_items else 0.0

    training_summary = {
        "model_name": MODEL_NAME,
        "model_type": "event_weighted_global_popularity",
        "top_k": TOP_K,
        "test_window_days": TEST_WINDOW_DAYS,
        "evaluation_sample_users": EVALUATION_SAMPLE_USERS,
        "min_timestamp_ms": min_timestamp_ms,
        "max_timestamp_ms": max_timestamp_ms,
        "cutoff_timestamp_ms": cutoff_timestamp_ms,
        "min_event_datetime": min_datetime,
        "max_event_datetime": max_datetime,
        "train_event_count": train_event_count,
        "test_event_count": test_event_count,
        "train_distinct_items": train_items,
        "model_saved_item_count": saved_model_items,
        "model_item_coverage": model_item_coverage,
        "test_target_users": int(len(test_targets_df)),
        **metrics,
    }

    save_model_artifact(
        item_scores_df=item_scores_df,
        training_summary=training_summary,
    )

    item_scores_df.head(1000).to_csv(POPULARITY_SCORES_PATH, index=False)
    recommendation_examples_df.to_csv(RECOMMENDATION_EXAMPLES_PATH, index=False)
    pd.DataFrame([training_summary]).to_csv(TRAINING_SUMMARY_PATH, index=False)

    run_id = log_mlflow(training_summary)

    print("Retailrocket popularity recommender training completed.")
    print(f"Model saved: {MODEL_PATH}")
    print(f"Training summary saved: {TRAINING_SUMMARY_PATH}")
    print(f"Recommendation examples saved: {RECOMMENDATION_EXAMPLES_PATH}")
    print(f"Popularity scores saved: {POPULARITY_SCORES_PATH}")
    print(f"MLflow experiment: {EXPERIMENT_NAME}")
    print(f"MLflow run_id: {run_id}")
    print()
    print(pd.DataFrame([training_summary]).to_string(index=False))


if __name__ == "__main__":
    main()
