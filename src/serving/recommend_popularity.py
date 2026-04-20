import argparse
import pickle
from pathlib import Path
from typing import Any

import pandas as pd


MODEL_PATH = Path("models/dummyjson/popularity_recommender.pkl")
RECOMMENDATION_OUTPUT_PATH = Path("reports/inference_demo_recommendations.csv")
SUMMARY_OUTPUT_PATH = Path("reports/inference_demo_summary.csv")


def load_model(model_path: Path) -> dict[str, Any]:
    """
    Loads the trained popularity recommender model artifact.

    The artifact was created by:
    src/training/train_popularity_recommender.py
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {model_path}. "
            "Run python -m src.training.train_popularity_recommender first, "
            "or restore the model with DVC."
        )

    with model_path.open("rb") as file:
        model = pickle.load(file)

    return model


def parse_demo_users(demo_users: str) -> list[int]:
    """
    Converts a comma-separated user string into a list of integers.

    Example:
    '1,2,3' -> [1, 2, 3]
    """
    users = []

    for value in demo_users.split(","):
        value = value.strip()

        if not value:
            continue

        users.append(int(value))

    return users


def get_seen_items(model: dict[str, Any], user_id: int) -> set[int]:
    """
    Gets items already seen by a user in the training data.

    We exclude these items from recommendations to avoid recommending
    products the user has already interacted with.
    """
    history = model.get("train_user_item_history", {})

    # Pickle may preserve integer keys, but this fallback also handles string keys.
    seen_items = history.get(user_id, history.get(str(user_id), []))

    return set(int(item_id) for item_id in seen_items)


def recommend_for_user(
    model: dict[str, Any],
    user_id: int,
    top_k: int,
) -> list[dict[str, Any]]:
    """
    Generates Top-K recommendations for one user using the global popularity model.
    """
    item_scores = model["item_scores"]
    seen_items = get_seen_items(model, user_id)

    recommendations = []

    for item in item_scores:
        item_id = int(item["item_id"])

        if item_id in seen_items:
            continue

        recommendations.append(
            {
                "user_id": user_id,
                "recommendation_rank": len(recommendations) + 1,
                "item_id": item_id,
                "title": item.get("title", "unknown"),
                "category": item.get("category", "unknown"),
                "brand": item.get("brand", "unknown"),
                "price": item.get("price"),
                "rating": item.get("rating"),
                "popularity_score": item.get("popularity_score"),
                "item_interaction_count": item.get("item_interaction_count"),
                "item_distinct_user_count": item.get("item_distinct_user_count"),
                "item_total_quantity": item.get("item_total_quantity"),
                "item_total_revenue": item.get("item_total_revenue"),
                "was_seen_in_training": False,
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
    rows = []

    for user_id in user_ids:
        user_recommendations = recommend_for_user(
            model=model,
            user_id=user_id,
            top_k=top_k,
        )

        rows.extend(user_recommendations)

    return pd.DataFrame(rows)


def build_summary(
    model: dict[str, Any],
    user_ids: list[int],
    recommendations_df: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    model_metrics = model.get("metrics", {})

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
        "training_hit_rate_at_10": model_metrics.get("hit_rate_at_10"),
        "training_precision_at_10": model_metrics.get("precision_at_10"),
        "training_recall_at_10": model_metrics.get("recall_at_10"),
        "catalog_item_coverage": model_metrics.get("catalog_item_coverage"),
        "interaction_item_coverage": model_metrics.get("interaction_item_coverage"),
    }

    return pd.DataFrame([summary])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate recommendations using the DummyJSON popularity recommender."
    )

    parser.add_argument(
        "--user-id",
        type=int,
        default=None,
        help="Single user ID for recommendation generation.",
    )

    parser.add_argument(
        "--demo-users",
        type=str,
        default="1,2,3,4,5",
        help="Comma-separated user IDs for demo recommendation generation.",
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
    else:
        user_ids = parse_demo_users(args.demo_users)

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

    print("Popularity recommendation inference completed.")
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