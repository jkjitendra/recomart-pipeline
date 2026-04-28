from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


WAREHOUSE_PATH = Path("data/warehouse/recomart.duckdb")

MODEL_SPECS = {
    "popularity": {
        "path": Path("models/retailrocket/popularity_recommender.pkl"),
        "training_command": "python -m src.training.train_retailrocket_popularity_recommender",
    },
    "content_based": {
        "path": Path("models/retailrocket/content_based_recommender.pkl"),
        "training_command": "python -m src.training.train_retailrocket_content_recommender",
    },
}

RECOMMENDATION_OUTPUT_PATH = Path("reports/retailrocket_inference_demo_recommendations.csv")
SUMMARY_OUTPUT_PATH = Path("reports/retailrocket_inference_demo_summary.csv")


def load_model(model_key: str) -> tuple[Path, dict[str, Any]]:
    model_spec = MODEL_SPECS[model_key]
    model_path = model_spec["path"]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {model_path}. "
            f"Run {model_spec['training_command']} first, or restore the model with DVC."
        )

    with model_path.open("rb") as file:
        model = pickle.load(file)

    return model_path, model


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
        history.setdefault(user_id, set()).add(item_id)

    return history


def get_user_content_profiles(user_ids: list[int]) -> dict[int, dict[str, dict[int, float]]]:
    if not user_ids:
        return {}

    users_df = pd.DataFrame({"user_id": user_ids})

    with duckdb.connect(str(WAREHOUSE_PATH)) as connection:
        connection.register("users_df", users_df)

        profile_df = connection.execute(
            """
            SELECT
                i.user_id,
                COALESCE(f.category_id, -1) AS category_id,
                COALESCE(f.parent_category_id, -1) AS parent_category_id,
                SUM(
                    CASE i.event_type
                        WHEN 'transaction' THEN 5.0
                        WHEN 'addtocart' THEN 3.0
                        ELSE 1.0
                    END
                ) AS profile_weight
            FROM mart.retailrocket_interactions i
            LEFT JOIN mart.retailrocket_item_features f
                ON i.item_id = f.item_id
            INNER JOIN users_df users
                ON i.user_id = users.user_id
            GROUP BY
                i.user_id,
                COALESCE(f.category_id, -1),
                COALESCE(f.parent_category_id, -1)
            """
        ).fetchdf()

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


def recommend_popularity_for_user(
    model: dict[str, Any],
    user_id: int,
    seen_items: set[int],
    top_k: int,
) -> list[dict[str, Any]]:
    recommendations = []

    for item in model.get("item_scores", []):
        item_id = int(item["item_id"])

        if item_id in seen_items:
            continue

        recommendations.append(
            {
                "user_id": user_id,
                "recommendation_rank": len(recommendations) + 1,
                "item_id": item_id,
                "recommendation_score": item.get("popularity_score"),
                "popularity_score": item.get("popularity_score"),
                "content_score": None,
                "category_score": None,
                "parent_category_score": None,
                "metadata_score": None,
                "conversion_score": None,
                "popularity_prior": None,
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


def recommend_content_for_user(
    model: dict[str, Any],
    user_id: int,
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

    for item in model.get("candidate_items", []):
        item_id = int(item["item_id"])

        if item_id in seen_items:
            continue

        category_id = int(item.get("category_id", -1))
        parent_category_id = int(item.get("parent_category_id", -1))

        category_score = category_weights.get(category_id, 0.0)
        parent_category_score = parent_category_weights.get(parent_category_id, 0.0)
        train_interaction_score = float(item.get("train_interaction_score", 0.0))

        metadata_score = (
            0.10 * float(item.get("is_available", 0.0))
            + 0.05 * float(item.get("has_category_metadata", 0.0))
            + 0.05 * float(item.get("has_availability_metadata", 0.0))
        )

        conversion_score = (
            0.50 * float(item.get("addtocart_rate", 0.0))
            + 1.00 * float(item.get("transaction_rate", 0.0))
        )

        popularity_prior = 0.05 * math.log1p(train_interaction_score)
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
                "recommendation_score": content_score,
                "popularity_score": item.get("global_content_prior_score"),
                "content_score": content_score,
                "category_score": category_score,
                "parent_category_score": parent_category_score,
                "metadata_score": metadata_score,
                "conversion_score": conversion_score,
                "popularity_prior": popularity_prior,
                "train_event_count": item.get("train_event_count"),
                "train_unique_users": item.get("train_unique_users"),
                "train_view_events": item.get("train_view_events"),
                "train_addtocart_events": item.get("train_addtocart_events"),
                "train_transaction_events": item.get("train_transaction_events"),
                "category_id": category_id,
                "parent_category_id": parent_category_id,
                "is_available": item.get("is_available"),
                "has_category_metadata": item.get("has_category_metadata"),
                "has_availability_metadata": item.get("has_availability_metadata"),
                "was_seen_before": False,
            }
        )

    scored_items.sort(
        key=lambda row: (
            float(row["content_score"]),
            float(row.get("train_event_count") or 0.0),
            int(row["item_id"]),
        ),
        reverse=True,
    )

    recommendations = scored_items[:top_k]

    for rank, item in enumerate(recommendations, start=1):
        item["recommendation_rank"] = rank

    return recommendations


def build_recommendations(
    model_key: str,
    model: dict[str, Any],
    user_ids: list[int],
    top_k: int,
) -> pd.DataFrame:
    seen_items_by_user = get_seen_items_for_users(user_ids)
    user_profiles = (
        get_user_content_profiles(user_ids)
        if model_key == "content_based"
        else {}
    )

    rows = []

    for user_id in user_ids:
        seen_items = seen_items_by_user.get(user_id, set())

        if model_key == "popularity":
            recommendations = recommend_popularity_for_user(
                model=model,
                user_id=user_id,
                seen_items=seen_items,
                top_k=top_k,
            )
        else:
            recommendations = recommend_content_for_user(
                model=model,
                user_id=user_id,
                user_profiles=user_profiles,
                seen_items=seen_items,
                top_k=top_k,
            )

        for recommendation in recommendations:
            recommendation["model_key"] = model_key
            recommendation["model_name"] = model.get("model_name")
            recommendation["model_type"] = model.get("model_type")

        rows.extend(recommendations)

    return pd.DataFrame(rows)


def build_summary(
    model_key: str,
    model_path: Path,
    model: dict[str, Any],
    user_ids: list[int],
    recommendations_df: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    metrics = model.get("training_summary") or model.get("metrics", {})

    summary = {
        "model_key": model_key,
        "model_name": model.get("model_name"),
        "model_type": model.get("model_type"),
        "model_artifact_path": str(model_path),
        "requested_user_count": len(user_ids),
        "top_k": top_k,
        "recommendation_rows": len(recommendations_df),
        "distinct_recommended_items": (
            int(recommendations_df["item_id"].nunique())
            if not recommendations_df.empty
            else 0
        ),
        "candidate_or_saved_item_count": (
            model.get("candidate_item_count")
            or metrics.get("candidate_item_count")
            or metrics.get("model_saved_item_count")
        ),
        "training_hit_rate_at_10": metrics.get("hit_rate_at_10"),
        "training_precision_at_10": metrics.get("precision_at_10"),
        "training_recall_at_10": metrics.get("recall_at_10"),
        "training_ndcg_at_10": metrics.get("ndcg_at_10"),
        "train_event_count": metrics.get("train_event_count"),
        "test_event_count": metrics.get("test_event_count"),
        "test_target_users": metrics.get("test_target_users"),
    }

    return pd.DataFrame([summary])


def write_merged_model_report(
    path: Path,
    current_df: pd.DataFrame,
    model_key: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    current_df = current_df.dropna(axis=1, how="all")

    if path.exists():
        existing_df = pd.read_csv(path)

        if "model_key" in existing_df.columns:
            existing_df = existing_df[
                existing_df["model_key"].notna()
                & (existing_df["model_key"].astype(str).str.strip() != "")
            ]
            existing_df = existing_df[existing_df["model_key"] != model_key]
        else:
            existing_df = existing_df.iloc[0:0]

        existing_df = existing_df.dropna(axis=1, how="all")

        if existing_df.empty:
            output_df = current_df
        else:
            output_df = pd.DataFrame(
                existing_df.to_dict(orient="records")
                + current_df.to_dict(orient="records")
            )
    else:
        output_df = current_df

    output_df.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate recommendations using Retailrocket recommender models."
    )

    parser.add_argument(
        "--model",
        choices=sorted(MODEL_SPECS.keys()),
        default="popularity",
        help="Retailrocket model artifact to use for recommendation generation.",
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

    model_path, model = load_model(args.model)

    if args.user_id is not None:
        user_ids = [args.user_id]
    elif args.demo_users is not None:
        user_ids = parse_demo_users(args.demo_users)
    else:
        user_ids = get_default_demo_users(limit=5)

    recommendations_df = build_recommendations(
        model_key=args.model,
        model=model,
        user_ids=user_ids,
        top_k=args.top_k,
    )

    summary_df = build_summary(
        model_key=args.model,
        model_path=model_path,
        model=model,
        user_ids=user_ids,
        recommendations_df=recommendations_df,
        top_k=args.top_k,
    )

    write_merged_model_report(
        path=RECOMMENDATION_OUTPUT_PATH,
        current_df=recommendations_df,
        model_key=args.model,
    )
    write_merged_model_report(
        path=SUMMARY_OUTPUT_PATH,
        current_df=summary_df,
        model_key=args.model,
    )

    print("Retailrocket recommendation inference completed.")
    print(f"Model selected: {args.model}")
    print(f"Model loaded: {model_path}")
    print(f"Users scored: {user_ids}")
    print(f"Top-K: {args.top_k}")
    print(f"Recommendations saved: {RECOMMENDATION_OUTPUT_PATH}")
    print(f"Summary saved: {SUMMARY_OUTPUT_PATH}")
    print()

    print("Current inference summary:")
    print(summary_df.to_string(index=False))
    print()

    print("Current recommendations:")
    print(recommendations_df.to_string(index=False))


if __name__ == "__main__":
    main()
