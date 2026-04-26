from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = PROJECT_ROOT / "configs/feature_store/retailrocket_feature_registry.json"
REGISTRY_SUMMARY_PATH = PROJECT_ROOT / "reports/feature_store_registry_summary.csv"
RETRIEVAL_DEMO_PATH = PROJECT_ROOT / "reports/feature_store_retrieval_demo.csv"
FEATURE_METADATA_PATH = PROJECT_ROOT / "reports/feature_metadata_documentation.csv"


ENTITY_COLUMNS = {"user_id", "item_id"}
METADATA_COLUMNS = {
    "category_id",
    "parent_category_id",
    "is_available",
    "available_raw_value",
    "has_category_metadata",
    "has_availability_metadata",
    "first_event_timestamp_ms",
    "last_event_timestamp_ms",
    "first_event_datetime",
    "last_event_datetime",
    "strongest_event_type",
}


def display_path(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)


def load_registry() -> dict[str, Any]:
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Feature registry not found: {display_path(REGISTRY_PATH)}")

    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def get_feature_view(registry: dict[str, Any], feature_view_name: str) -> dict[str, Any]:
    for feature_view in registry["feature_views"]:
        if feature_view["name"] == feature_view_name:
            return feature_view

    raise ValueError(f"Unknown feature view: {feature_view_name}")


def source_path(feature_view: dict[str, Any]) -> Path:
    return PROJECT_ROOT / feature_view["source_path"]


def read_schema(path: Path) -> list[tuple[str, str]]:
    schema = pq.read_schema(path)
    return [(field.name, str(field.type)) for field in schema]


def classify_feature_column(column_name: str, entity_keys: list[str]) -> str:
    if column_name in entity_keys or column_name in ENTITY_COLUMNS and column_name in entity_keys:
        return "entity_key"

    if column_name in METADATA_COLUMNS:
        return "metadata"

    return "feature"


def build_registry_summary(registry: dict[str, Any]) -> pd.DataFrame:
    rows = []

    for feature_view in registry["feature_views"]:
        path = source_path(feature_view)
        source_exists = path.exists()
        schema = read_schema(path) if source_exists else []
        feature_names = [
            column_name
            for column_name, _ in schema
            if column_name not in set(feature_view["entity_keys"])
        ]
        data_types = {column_name: data_type for column_name, data_type in schema}
        feature_roles = {
            column_name: classify_feature_column(column_name, feature_view["entity_keys"])
            for column_name, _ in schema
        }

        rows.append(
            {
                "registry_name": registry["registry_name"],
                "registry_version": registry["registry_version"],
                "feature_view_name": feature_view["name"],
                "feature_view_version": feature_view["version"],
                "entity_keys": json.dumps(feature_view["entity_keys"]),
                "source_path": feature_view["source_path"],
                "source_table": feature_view.get("source_table", ""),
                "source_exists": source_exists,
                "feature_count": len(feature_names),
                "feature_names": ", ".join(feature_names),
                "data_types": json.dumps(data_types, sort_keys=True),
                "feature_roles": json.dumps(feature_roles, sort_keys=True),
                "transformation": feature_view["transformation"],
                "used_for": ", ".join(feature_view["used_for"]),
                "created_by": feature_view["created_by"],
                "description": feature_view["description"],
            }
        )

    return pd.DataFrame(rows)


def build_feature_metadata_documentation(registry: dict[str, Any]) -> pd.DataFrame:
    rows = []

    for feature_view in registry["feature_views"]:
        path = source_path(feature_view)
        source_exists = path.exists()
        schema = read_schema(path) if source_exists else []

        if not schema:
            rows.append(
                {
                    "registry_name": registry["registry_name"],
                    "registry_version": registry["registry_version"],
                    "feature_view_name": feature_view["name"],
                    "feature_view_version": feature_view["version"],
                    "entity_keys": json.dumps(feature_view["entity_keys"]),
                    "source_path": feature_view["source_path"],
                    "source_table": feature_view.get("source_table", ""),
                    "source_exists": source_exists,
                    "column_name": "",
                    "data_type": "",
                    "feature_role": "missing_source_file",
                    "transformation": feature_view["transformation"],
                    "used_for": ", ".join(feature_view["used_for"]),
                    "created_by": feature_view["created_by"],
                    "description": feature_view["description"],
                }
            )
            continue

        for column_name, data_type in schema:
            rows.append(
                {
                    "registry_name": registry["registry_name"],
                    "registry_version": registry["registry_version"],
                    "feature_view_name": feature_view["name"],
                    "feature_view_version": feature_view["version"],
                    "entity_keys": json.dumps(feature_view["entity_keys"]),
                    "source_path": feature_view["source_path"],
                    "source_table": feature_view.get("source_table", ""),
                    "source_exists": source_exists,
                    "column_name": column_name,
                    "data_type": data_type,
                    "feature_role": classify_feature_column(column_name, feature_view["entity_keys"]),
                    "transformation": feature_view["transformation"],
                    "used_for": ", ".join(feature_view["used_for"]),
                    "created_by": feature_view["created_by"],
                    "description": feature_view["description"],
                }
            )

    return pd.DataFrame(rows)


def read_feature_view(feature_view: dict[str, Any], columns: list[str] | None = None) -> pd.DataFrame:
    path = source_path(feature_view)
    if not path.exists():
        raise FileNotFoundError(f"Feature source not found: {display_path(path)}")

    return pd.read_parquet(path, columns=columns)


def choose_sample_user_item(registry: dict[str, Any]) -> tuple[int, int]:
    user_item_view = get_feature_view(registry, "retailrocket_user_item_features")
    df = read_feature_view(
        user_item_view,
        columns=["user_id", "item_id", "interaction_score", "has_transaction", "has_addtocart"],
    )

    df = df.sort_values(
        ["has_transaction", "has_addtocart", "interaction_score", "user_id", "item_id"],
        ascending=[False, False, False, True, True],
    )
    row = df.iloc[0]
    return int(row["user_id"]), int(row["item_id"])


def retrieve_user_features(registry: dict[str, Any], user_id: int) -> pd.DataFrame:
    feature_view = get_feature_view(registry, "retailrocket_user_features")
    df = read_feature_view(feature_view)
    return df[df["user_id"] == user_id].head(1).copy()


def retrieve_item_features(registry: dict[str, Any], item_id: int) -> pd.DataFrame:
    feature_view = get_feature_view(registry, "retailrocket_item_features")
    df = read_feature_view(feature_view)
    return df[df["item_id"] == item_id].head(1).copy()


def retrieve_user_item_features(registry: dict[str, Any], user_id: int, item_id: int) -> pd.DataFrame:
    feature_view = get_feature_view(registry, "retailrocket_user_item_features")
    df = read_feature_view(feature_view)
    return df[(df["user_id"] == user_id) & (df["item_id"] == item_id)].head(1).copy()


def summarize_retrieval(
    *,
    retrieval_type: str,
    feature_view_name: str,
    entity_filter: str,
    df: pd.DataFrame,
) -> dict[str, Any]:
    sample_values = {}
    if not df.empty:
        row = df.iloc[0].to_dict()
        for key, value in row.items():
            sample_values[key] = str(value)

    return {
        "retrieval_type": retrieval_type,
        "feature_view_name": feature_view_name,
        "feature_view_version": "v1",
        "entity_filter": entity_filter,
        "rows_returned": len(df),
        "columns_returned": len(df.columns),
        "sample_values_json": json.dumps(sample_values, sort_keys=True),
    }


def build_retrieval_demo(registry: dict[str, Any]) -> pd.DataFrame:
    user_id, item_id = choose_sample_user_item(registry)

    user_features = retrieve_user_features(registry, user_id)
    item_features = retrieve_item_features(registry, item_id)
    user_item_features = retrieve_user_item_features(registry, user_id, item_id)

    rows = [
        summarize_retrieval(
            retrieval_type="user_features",
            feature_view_name="retailrocket_user_features",
            entity_filter=f"user_id={user_id}",
            df=user_features,
        ),
        summarize_retrieval(
            retrieval_type="item_features",
            feature_view_name="retailrocket_item_features",
            entity_filter=f"item_id={item_id}",
            df=item_features,
        ),
        summarize_retrieval(
            retrieval_type="user_item_features",
            feature_view_name="retailrocket_user_item_features",
            entity_filter=f"user_id={user_id},item_id={item_id}",
            df=user_item_features,
        ),
    ]

    return pd.DataFrame(rows)


def main() -> None:
    registry = load_registry()

    REGISTRY_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    registry_summary_df = build_registry_summary(registry)
    retrieval_demo_df = build_retrieval_demo(registry)
    metadata_df = build_feature_metadata_documentation(registry)

    registry_summary_df.to_csv(REGISTRY_SUMMARY_PATH, index=False)
    retrieval_demo_df.to_csv(RETRIEVAL_DEMO_PATH, index=False)
    metadata_df.to_csv(FEATURE_METADATA_PATH, index=False)

    print("Retailrocket feature store retrieval demo completed.")
    print(f"Registry path: {display_path(REGISTRY_PATH)}")
    print(f"Registry summary saved: {display_path(REGISTRY_SUMMARY_PATH)}")
    print(f"Retrieval demo saved: {display_path(RETRIEVAL_DEMO_PATH)}")
    print(f"Feature metadata documentation saved: {display_path(FEATURE_METADATA_PATH)}")
    print()
    print("Registry summary:")
    print(registry_summary_df[["feature_view_name", "entity_keys", "source_path", "source_exists", "feature_count"]].to_string(index=False))
    print()
    print("Retrieval demo:")
    print(retrieval_demo_df[["retrieval_type", "feature_view_name", "entity_filter", "rows_returned", "columns_returned"]].to_string(index=False))
    print()
    print("Feature metadata rows:", len(metadata_df))


if __name__ == "__main__":
    main()
