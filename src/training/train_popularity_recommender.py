import pickle
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd


INTERACTION_FEATURES_PATH = Path("data/features/source=dummyjson/interaction_features.parquet")
ITEM_FEATURES_PATH = Path("data/features/source=dummyjson/item_features.parquet")

MODEL_DIR = Path("models/dummyjson")
MODEL_PATH = MODEL_DIR / "popularity_recommender.pkl"

TRAINING_SUMMARY_PATH = Path("reports/model_training_summary.csv")
RECOMMENDATION_EXAMPLES_PATH = Path("reports/recommendation_examples.csv")
POPULARITY_SCORES_PATH = Path("reports/popularity_item_scores.csv")

EXPERIMENT_NAME = "dummyjson_recommendation_baselines"
MODEL_NAME = "popularity_recommender"
TOP_K = 10
RANDOM_STATE = 42


def load_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not INTERACTION_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Interaction features not found: {INTERACTION_FEATURES_PATH}. "
            "Run python -m src.transformation.build_dummyjson_features first."
        )

    if not ITEM_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Item features not found: {ITEM_FEATURES_PATH}. "
            "Run python -m src.transformation.build_dummyjson_features first."
        )

    interactions_df = pd.read_parquet(INTERACTION_FEATURES_PATH)
    items_df = pd.read_parquet(ITEM_FEATURES_PATH)

    return interactions_df, items_df


def create_holdout_split(
    interactions_df: pd.DataFrame,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates a simple per-user holdout split.

    For each user:
    - one interacted item is held out for testing
    - the remaining interactions are used for training

    This is a simple offline evaluation strategy for the baseline model.
    """
    rng = np.random.default_rng(random_state)

    test_indices = []

    for user_id, user_rows in interactions_df.groupby("user_id"):
        if len(user_rows) < 2:
            continue

        selected_index = rng.choice(user_rows.index.to_numpy())
        test_indices.append(selected_index)

    test_df = interactions_df.loc[test_indices].copy()
    train_df = interactions_df.drop(index=test_indices).copy()

    return train_df, test_df


def compute_popularity_scores(
    train_df: pd.DataFrame,
    items_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Computes item popularity using interaction count, distinct users,
    total quantity, and total revenue.

    The score is intentionally simple and explainable.
    """
    item_scores = (
        train_df.groupby("item_id")
        .agg(
            item_interaction_count=("item_id", "size"),
            item_distinct_user_count=("user_id", "nunique"),
            item_total_quantity=("interaction_weight", "sum"),
            item_total_revenue=("line_total", "sum"),
            item_avg_quantity=("interaction_weight", "mean"),
            item_avg_line_total=("line_total", "mean"),
            has_catalog_metadata=("has_catalog_metadata", "max"),
        )
        .reset_index()
    )

    # Normalize score components so no single large-value column dominates completely.
    item_scores["interaction_component"] = item_scores["item_interaction_count"]
    item_scores["user_component"] = item_scores["item_distinct_user_count"]
    item_scores["quantity_component"] = np.log1p(item_scores["item_total_quantity"])
    item_scores["revenue_component"] = np.log1p(item_scores["item_total_revenue"].clip(lower=0))

    item_scores["popularity_score"] = (
        0.40 * item_scores["interaction_component"]
        + 0.25 * item_scores["user_component"]
        + 0.20 * item_scores["quantity_component"]
        + 0.15 * item_scores["revenue_component"]
    )

    item_metadata = items_df[
        [
            "item_id",
            "title",
            "category",
            "price",
            "rating",
            "stock",
            "brand",
        ]
    ].drop_duplicates(subset=["item_id"])

    item_scores = item_scores.merge(
        item_metadata,
        on="item_id",
        how="left",
    )

    item_scores["title"] = item_scores["title"].fillna("unknown")
    item_scores["category"] = item_scores["category"].fillna("unknown")
    item_scores["brand"] = item_scores["brand"].fillna("unknown")

    item_scores = item_scores.sort_values(
        by=["popularity_score", "item_interaction_count", "item_total_quantity"],
        ascending=False,
    ).reset_index(drop=True)

    item_scores["rank"] = np.arange(1, len(item_scores) + 1)

    return item_scores


def recommend_for_user(
    user_id: int,
    train_df: pd.DataFrame,
    item_scores: pd.DataFrame,
    top_k: int = TOP_K,
) -> list[int]:
    """
    Recommends top popular items that the user has not already interacted with
    in the training data.
    """
    seen_items = set(train_df.loc[train_df["user_id"] == user_id, "item_id"].tolist())

    recommendations = (
        item_scores.loc[~item_scores["item_id"].isin(seen_items), "item_id"]
        .head(top_k)
        .astype(int)
        .tolist()
    )

    return recommendations


def evaluate_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    item_scores: pd.DataFrame,
    top_k: int = TOP_K,
) -> dict[str, Any]:
    """
    Evaluates recommendations using simple holdout metrics.

    Since each test user has one held-out item:
    - HitRate@K = fraction of users where held-out item appears in Top-K
    - Recall@K = same as HitRate@K in this one-item holdout setup
    - Precision@K = hits / total recommended slots
    """
    if test_df.empty:
        return {
            "evaluated_users": 0,
            "hits": 0,
            f"hit_rate_at_{top_k}": 0.0,
            f"precision_at_{top_k}": 0.0,
            f"recall_at_{top_k}": 0.0,
        }

    hits = 0
    evaluated_users = 0

    for _, row in test_df.iterrows():
        user_id = int(row["user_id"])
        true_item_id = int(row["item_id"])

        recommendations = recommend_for_user(
            user_id=user_id,
            train_df=train_df,
            item_scores=item_scores,
            top_k=top_k,
        )

        if not recommendations:
            continue

        evaluated_users += 1

        if true_item_id in recommendations:
            hits += 1

    hit_rate = hits / evaluated_users if evaluated_users else 0.0
    precision = hits / (evaluated_users * top_k) if evaluated_users else 0.0
    recall = hit_rate

    return {
        "evaluated_users": evaluated_users,
        "hits": hits,
        f"hit_rate_at_{top_k}": hit_rate,
        f"precision_at_{top_k}": precision,
        f"recall_at_{top_k}": recall,
    }


def build_recommendation_examples(
    train_df: pd.DataFrame,
    item_scores: pd.DataFrame,
    users: list[int],
    top_k: int = 5,
) -> pd.DataFrame:
    rows = []

    item_lookup = item_scores.set_index("item_id").to_dict(orient="index")

    for user_id in users:
        recommendations = recommend_for_user(
            user_id=user_id,
            train_df=train_df,
            item_scores=item_scores,
            top_k=top_k,
        )

        for rank, item_id in enumerate(recommendations, start=1):
            metadata = item_lookup.get(item_id, {})

            rows.append(
                {
                    "user_id": user_id,
                    "recommendation_rank": rank,
                    "item_id": item_id,
                    "title": metadata.get("title", "unknown"),
                    "category": metadata.get("category", "unknown"),
                    "brand": metadata.get("brand", "unknown"),
                    "popularity_score": metadata.get("popularity_score"),
                    "item_interaction_count": metadata.get("item_interaction_count"),
                    "item_total_quantity": metadata.get("item_total_quantity"),
                }
            )

    return pd.DataFrame(rows)


def save_model_artifact(
    item_scores: pd.DataFrame,
    train_df: pd.DataFrame,
    metrics: dict[str, Any],
) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_artifact = {
        "model_name": MODEL_NAME,
        "model_type": "global_popularity_baseline",
        "top_items": item_scores["item_id"].astype(int).tolist(),
        "item_scores": item_scores.to_dict(orient="records"),
        "train_user_item_history": (
            train_df.groupby("user_id")["item_id"]
            .apply(lambda values: sorted(set(int(value) for value in values)))
            .to_dict()
        ),
        "metrics": metrics,
    }

    with MODEL_PATH.open("wb") as file:
        pickle.dump(model_artifact, file)


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    interactions_df, items_df = load_features()

    train_df, test_df = create_holdout_split(interactions_df)
    item_scores = compute_popularity_scores(train_df, items_df)

    metrics = evaluate_model(train_df, test_df, item_scores, top_k=TOP_K)

    unique_users = int(interactions_df["user_id"].nunique())
    unique_items = int(interactions_df["item_id"].nunique())
    train_interactions = int(len(train_df))
    test_interactions = int(len(test_df))
    catalog_items = int(items_df["item_id"].nunique())
    recommended_items = int(item_scores["item_id"].nunique())

    catalog_item_ids = set(items_df["item_id"].astype(int).tolist())
    recommended_item_ids = set(item_scores["item_id"].astype(int).tolist())
    recommended_catalog_item_count = len(recommended_item_ids & catalog_item_ids)

    catalog_item_coverage = (
        recommended_catalog_item_count / catalog_items
        if catalog_items
        else 0.0
    )

    interaction_item_coverage = (
        recommended_items / unique_items
        if unique_items
        else 0.0
    )

    training_summary = {
        "model_name": MODEL_NAME,
        "model_type": "global_popularity_baseline",
        "top_k": TOP_K,
        "total_interactions": int(len(interactions_df)),
        "train_interactions": train_interactions,
        "test_interactions": test_interactions,
        "unique_users": unique_users,
        "unique_items_in_interactions": unique_items,
        "catalog_items": catalog_items,
        "recommended_item_count": recommended_items,
        "recommended_catalog_item_count": recommended_catalog_item_count,
        "catalog_item_coverage": catalog_item_coverage,
        "interaction_item_coverage": interaction_item_coverage,
        **metrics,
    }

    save_model_artifact(item_scores, train_df, training_summary)

    item_scores.to_csv(POPULARITY_SCORES_PATH, index=False)

    sample_users = sorted(interactions_df["user_id"].drop_duplicates().astype(int).head(5).tolist())
    recommendation_examples_df = build_recommendation_examples(
        train_df=train_df,
        item_scores=item_scores,
        users=sample_users,
        top_k=5,
    )
    recommendation_examples_df.to_csv(RECOMMENDATION_EXAMPLES_PATH, index=False)

    summary_df = pd.DataFrame([training_summary])
    summary_df.to_csv(TRAINING_SUMMARY_PATH, index=False)

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=MODEL_NAME) as run:
        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_param("model_type", "global_popularity_baseline")
        mlflow.log_param("top_k", TOP_K)
        mlflow.log_param("random_state", RANDOM_STATE)

        mlflow.log_metric("total_interactions", int(len(interactions_df)))
        mlflow.log_metric("train_interactions", train_interactions)
        mlflow.log_metric("test_interactions", test_interactions)
        mlflow.log_metric("unique_users", unique_users)
        mlflow.log_metric("unique_items_in_interactions", unique_items)
        mlflow.log_metric("catalog_items", catalog_items)
        mlflow.log_metric("recommended_item_count", recommended_items)
        mlflow.log_metric("recommended_catalog_item_count", recommended_catalog_item_count)
        mlflow.log_metric("catalog_item_coverage", catalog_item_coverage)
        mlflow.log_metric("interaction_item_coverage", interaction_item_coverage)

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, float(metric_value))

        mlflow.log_artifact(str(MODEL_PATH))
        mlflow.log_artifact(str(TRAINING_SUMMARY_PATH))
        mlflow.log_artifact(str(RECOMMENDATION_EXAMPLES_PATH))
        mlflow.log_artifact(str(POPULARITY_SCORES_PATH))

        run_id = run.info.run_id

    print("Popularity recommender training completed.")
    print(f"Model saved: {MODEL_PATH}")
    print(f"Training summary saved: {TRAINING_SUMMARY_PATH}")
    print(f"Recommendation examples saved: {RECOMMENDATION_EXAMPLES_PATH}")
    print(f"Popularity scores saved: {POPULARITY_SCORES_PATH}")
    print(f"MLflow experiment: {EXPERIMENT_NAME}")
    print(f"MLflow run_id: {run_id}")
    print()
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()