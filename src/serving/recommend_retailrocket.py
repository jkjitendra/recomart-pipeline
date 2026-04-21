import argparse
import pickle
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


WAREHOUSE_PATH = Path("data/warehouse/recommart.duckdb")
MODEL_PATH = Path("models/retailrocket/popularity_recommender.pkl")

RECOMMENDATION_OUTPUT_PATH = Path("reports/retailrocket_inference_demo_recommendations.csv")
SUMMARY_OUTPUT_PATH = Path("reports/retailrocket_inference_demo_summary.csv")


def load_model(model_path: Path) -> dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {model_path}. "
            "Run python -m src.training.train_retailrocket_popularity_recommender first, "
            "or restore the model with DVC."
        )

    with model_path.open("rb") as file:
        model = pickle.load(file)

    return model


def parse_demo_users(demo_users: str) -> list[int]:
    users = []

    for value in demo_users.split(","):
        value = value.strip()

        if value:
            users.append(int(value))

    return users


def get_default_demo_users(limit: int = 5) -> list[int]:
    """
    Picks active users from the Retailrocket warehouse.

    We choose users with at least one add-to-cart or transaction event,
    because recommendation examples are more meaningful for active users.
    """
    if not WAREHOUSE_PATH.exists():
        raise FileNotFoundError(
            f"Warehouse not found: {WAREHOUSE_PATH}. "
            "Run python -m src.transformation.load_retailrocket_to_duckdb first."
        )

    query = f"""
    SELECT
        user_id
    FROM features.retailrocket_user_features
    WHERE addtocart_events > 0
       OR transaction_events > 0
    ORDER BY total_interaction_score DESC, total_events DESC
    LIMIT {limit}
    """

    with duckdb.connect(str(WAREHOUSE_PATH)) as connection:
        users_df = connection.execute(query).fetchdf()

    return users_df["user_id"].astype(int).tolist()


def get_seen_items_for_users(user_ids: list[int]) -> dict[int, set[int]]:
    """
    Gets all historical items already seen by users.

    For inference demo, we exclude previously interacted items so that the
    recommendations are not simply products the user already viewed/carted/bought.
    """
    if not user_ids:
        return {}

    users_df = pd.DataFrame({"user_id": user_ids})

    with duckdb.connect(str(WAREHOUSE_PATH)) as connection:
        connection.register("users_df", users_df)

        seen_df = connection.execute(
            """
            SELECT DISTINCT
                ui.user_id,
                ui.item_id
            FROM features.retailrocket_user_item_features ui
            INNER JOIN users_df users
                ON ui.user_id = users.user_id
            """
        ).fetchdf()

    history: dict[int, set[int]] = {}

    for row in seen_df.itertuples(index=False):
        user_id = int(row.user_id)
        item_id = int(row.item_id)

        if user_id not in history:
            history[user_id] = set()

        history[user_id].add(item_id)

    return history


def recommend_for_user(
    model: dict[str, Any],
    user_id: int,
    seen_items: set[int],
    top_k: int,
) -> list[dict[str, Any]]:
    recommendations = []

    item_scores = model.get("item_scores", [])

    for item in item_scores:
        item_id = int(item["item_id"])

        if item_id in seen_items:
            continue

        recommendations.append(
            {
                "user_id": user_id,
                "recommendation_rank": len(recommendations) + 1,
                "item_id": item_id,
                "popularity_score": item.get("popularity_score"),
                "train_event_count": item.get("train_event_count"),
                "train_unique_users": item.get("train_unique_users"),
                "train_view_events": item.get("train_view_events"),
                "train_addtocart_events": item.get("train_addtocart_events"),
                "train_transaction_events": item.get("train_transaction_events"),
                "category_id": item.get("category_id"),
                "parent_category_id": item.get("parent_category_id"),
                "is_available": item.get("is_available"),
                "has_category_metadata": item.get("has_category_metadata"),
                "has_availability_metadata": item.get("has_availability_metadata"),
                "was_seen_before": False,
            }
        )

        if len(recommendations) >= top_k:
            break

    return recommendations


def build_recommendations(
    model: dict[str, Any],
    user_ids: list[int],
    top_k: int,
) -> pd.DataFrame:
    seen_items_by_user = get_seen_items_for_users(user_ids)

    rows = []

    for user_id in user_ids:
        seen_items = seen_items_by_user.get(user_id, set())

        rows.extend(
            recommend_for_user(
                model=model,
                user_id=user_id,
                seen_items=seen_items,
                top_k=top_k,
            )
        )

    return pd.DataFrame(rows)


def build_summary(
    model: dict[str, Any],
    user_ids: list[int],
    recommendations_df: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    metrics = model.get("metrics", {})

    summary = {
        "model_name": model.get("model_name"),
        "model_type": model.get("model_type"),
        "requested_user_count": len(user_ids),
        "top_k": top_k,
        "recommendation_rows": len(recommendations_df),
        "distinct_recommended_items": (
            int(recommendations_df["item_id"].nunique())
            if not recommendations_df.empty
            else 0
        ),
        "model_saved_item_count": metrics.get("model_saved_item_count"),
        "training_hit_rate_at_10": metrics.get("hit_rate_at_10"),
        "training_precision_at_10": metrics.get("precision_at_10"),
        "training_recall_at_10": metrics.get("recall_at_10"),
        "train_event_count": metrics.get("train_event_count"),
        "test_event_count": metrics.get("test_event_count"),
        "test_target_users": metrics.get("test_target_users"),
    }

    return pd.DataFrame([summary])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate recommendations using the Retailrocket popularity recommender."
    )

    parser.add_argument(
        "--user-id",
        type=int,
        default=None,
        help="Single Retailrocket visitor/user ID for recommendation generation.",
    )

    parser.add_argument(
        "--demo-users",
        type=str,
        default=None,
        help="Comma-separated Retailrocket user IDs. If omitted, active demo users are selected automatically.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of recommendations to return per user.",
    )

    args = parser.parse_args()

    model = load_model(MODEL_PATH)

    if args.user_id is not None:
        user_ids = [args.user_id]
    elif args.demo_users is not None:
        user_ids = parse_demo_users(args.demo_users)
    else:
        user_ids = get_default_demo_users(limit=5)

    recommendations_df = build_recommendations(
        model=model,
        user_ids=user_ids,
        top_k=args.top_k,
    )

    summary_df = build_summary(
        model=model,
        user_ids=user_ids,
        recommendations_df=recommendations_df,
        top_k=args.top_k,
    )

    RECOMMENDATION_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    recommendations_df.to_csv(RECOMMENDATION_OUTPUT_PATH, index=False)
    summary_df.to_csv(SUMMARY_OUTPUT_PATH, index=False)

    print("Retailrocket recommendation inference completed.")
    print(f"Model loaded: {MODEL_PATH}")
    print(f"Users scored: {user_ids}")
    print(f"Top-K: {args.top_k}")
    print(f"Recommendations saved: {RECOMMENDATION_OUTPUT_PATH}")
    print(f"Summary saved: {SUMMARY_OUTPUT_PATH}")
    print()

    print("Inference summary:")
    print(summary_df.to_string(index=False))
    print()

    print("Recommendations:")
    print(recommendations_df.to_string(index=False))


if __name__ == "__main__":
    main()