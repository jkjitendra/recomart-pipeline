import json
from pathlib import Path
from typing import Any

import pandas as pd


REGISTRY_PATH = Path("configs/feature_store/dummyjson_feature_registry.json")
REGISTRY_SUMMARY_PATH = Path("reports/feature_store_registry_summary.csv")
RETRIEVAL_DEMO_PATH = Path("reports/feature_store_retrieval_demo.csv")


def load_registry(registry_path: Path) -> dict[str, Any]:
    if not registry_path.exists():
        raise FileNotFoundError(f"Feature registry not found: {registry_path}")

    with registry_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_feature_view(registry: dict[str, Any], feature_view_name: str) -> dict[str, Any]:
    for feature_view in registry["feature_views"]:
        if feature_view["name"] == feature_view_name:
            return feature_view

    raise ValueError(f"Feature view not found in registry: {feature_view_name}")


def load_feature_view(registry: dict[str, Any], feature_view_name: str) -> pd.DataFrame:
    feature_view = get_feature_view(registry, feature_view_name)
    source_path = Path(feature_view["source_path"])

    if not source_path.exists():
        raise FileNotFoundError(
            f"Feature source file not found for {feature_view_name}: {source_path}"
        )

    return pd.read_parquet(source_path)


def retrieve_user_features(registry: dict[str, Any], user_id: int) -> pd.DataFrame:
    df = load_feature_view(registry, "dummyjson_user_features")
    return df[df["user_id"] == user_id].copy()


def retrieve_item_features(registry: dict[str, Any], item_id: int) -> pd.DataFrame:
    df = load_feature_view(registry, "dummyjson_item_features")
    return df[df["item_id"] == item_id].copy()


def retrieve_interaction_features(
    registry: dict[str, Any],
    user_id: int | None = None,
    item_id: int | None = None,
    limit: int = 10,
) -> pd.DataFrame:
    df = load_feature_view(registry, "dummyjson_interaction_features")

    if user_id is not None:
        df = df[df["user_id"] == user_id]

    if item_id is not None:
        df = df[df["item_id"] == item_id]

    return df.head(limit).copy()


def build_registry_summary(registry: dict[str, Any]) -> pd.DataFrame:
    rows = []

    for feature_view in registry["feature_views"]:
        source_path = Path(feature_view["source_path"])

        row = {
            "registry_name": registry["registry_name"],
            "registry_version": registry["registry_version"],
            "feature_view_name": feature_view["name"],
            "feature_view_version": feature_view["version"],
            "entity": json.dumps(feature_view["entity"]),
            "source_path": feature_view["source_path"],
            "source_exists": source_path.exists(),
            "feature_count": len(feature_view["features"]),
            "features": ", ".join(feature_view["features"]),
            "used_for": ", ".join(feature_view["used_for"]),
            "created_by": feature_view["created_by"],
        }

        rows.append(row)

    return pd.DataFrame(rows)


def build_retrieval_demo(registry: dict[str, Any]) -> pd.DataFrame:
    """
    Creates a small demonstration report showing feature retrieval results.

    This is not meant to be a model training dataset.
    It is evidence that the feature registry can retrieve user, item,
    and interaction features from versioned feature files.
    """
    demo_rows = []

    user_features = retrieve_user_features(registry, user_id=1)
    item_features = retrieve_item_features(registry, item_id=86)
    interaction_features = retrieve_interaction_features(
        registry,
        user_id=2,
        item_id=86,
        limit=5,
    )

    demo_rows.append(
        {
            "retrieval_type": "user_features",
            "entity_filter": "user_id=1",
            "rows_returned": len(user_features),
            "columns_returned": len(user_features.columns),
            "source_feature_view": "dummyjson_user_features",
        }
    )

    demo_rows.append(
        {
            "retrieval_type": "item_features",
            "entity_filter": "item_id=86",
            "rows_returned": len(item_features),
            "columns_returned": len(item_features.columns),
            "source_feature_view": "dummyjson_item_features",
        }
    )

    demo_rows.append(
        {
            "retrieval_type": "interaction_features",
            "entity_filter": "user_id=2,item_id=86",
            "rows_returned": len(interaction_features),
            "columns_returned": len(interaction_features.columns),
            "source_feature_view": "dummyjson_interaction_features",
        }
    )

    return pd.DataFrame(demo_rows)


def main() -> None:
    registry = load_registry(REGISTRY_PATH)

    REGISTRY_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    RETRIEVAL_DEMO_PATH.parent.mkdir(parents=True, exist_ok=True)

    registry_summary_df = build_registry_summary(registry)
    retrieval_demo_df = build_retrieval_demo(registry)

    registry_summary_df.to_csv(REGISTRY_SUMMARY_PATH, index=False)
    retrieval_demo_df.to_csv(RETRIEVAL_DEMO_PATH, index=False)

    print("Feature registry retrieval demo completed.")
    print(f"Registry path: {REGISTRY_PATH}")
    print(f"Registry summary saved: {REGISTRY_SUMMARY_PATH}")
    print(f"Retrieval demo saved: {RETRIEVAL_DEMO_PATH}")
    print()

    print("Registry summary:")
    print(registry_summary_df.to_string(index=False))
    print()

    print("Retrieval demo:")
    print(retrieval_demo_df.to_string(index=False))


if __name__ == "__main__":
    main()