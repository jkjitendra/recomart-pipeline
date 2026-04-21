from __future__ import annotations

import re
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq


WAREHOUSE_PATH = Path("data/warehouse/recomart.duckdb")
REPORTS_DIR = Path("reports")
PLOTS_DIR = REPORTS_DIR / "plots"

SQL_SCHEMA_SUMMARY_PATH = REPORTS_DIR / "sql_schema_summary.csv"
FEATURE_LOGIC_SUMMARY_PATH = REPORTS_DIR / "feature_logic_summary.csv"
FEATURE_METADATA_DOCUMENTATION_PATH = REPORTS_DIR / "feature_metadata_documentation.csv"
DVC_VERSIONING_SUMMARY_PATH = REPORTS_DIR / "dvc_versioning_summary.csv"
REPOSITORY_STRUCTURE_SUMMARY_PATH = REPORTS_DIR / "repository_structure_summary.csv"


def ensure_dirs() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def save_event_distribution_plot() -> None:
    input_path = REPORTS_DIR / "retailrocket_event_type_summary.csv"
    output_path = PLOTS_DIR / "retailrocket_event_distribution.png"

    if not input_path.exists():
        print(f"Skipping event distribution plot. Missing: {input_path}")
        return

    df = pd.read_csv(input_path)

    plt.figure(figsize=(8, 5))
    plt.bar(df["event"], df["row_count"])
    plt.title("Retailrocket Event Distribution")
    plt.xlabel("Event type")
    plt.ylabel("Row count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    print(f"Saved plot: {output_path}")


def save_top_items_plot() -> None:
    input_path = REPORTS_DIR / "retailrocket_popularity_item_scores.csv"
    output_path = PLOTS_DIR / "retailrocket_top_items.png"

    if not input_path.exists():
        print(f"Skipping top items plot. Missing: {input_path}")
        return

    df = pd.read_csv(input_path).head(20).copy()
    df["item_id"] = df["item_id"].astype(str)

    plt.figure(figsize=(11, 6))
    plt.bar(df["item_id"], df["popularity_score"])
    plt.title("Top 20 Retailrocket Items by Popularity Score")
    plt.xlabel("Item ID")
    plt.ylabel("Popularity score")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    print(f"Saved plot: {output_path}")


def save_user_activity_distribution_plot() -> None:
    input_path = Path("data/features/source=retailrocket/user_features.parquet")
    output_path = PLOTS_DIR / "retailrocket_user_activity_distribution.png"

    if not input_path.exists():
        print(f"Skipping user activity distribution plot. Missing: {input_path}")
        return

    df = pd.read_parquet(input_path, columns=["total_events"])

    if df.empty:
        print("Skipping user activity distribution plot. Empty user feature file.")
        return

    upper_clip = df["total_events"].quantile(0.99)
    clipped = df["total_events"].clip(upper=upper_clip)

    plt.figure(figsize=(8, 5))
    plt.hist(clipped, bins=50)
    plt.title("Retailrocket User Activity Distribution")
    plt.xlabel("Total events per user, clipped at 99th percentile")
    plt.ylabel("User count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    print(f"Saved plot: {output_path}")


def save_model_metrics_plot() -> None:
    input_path = REPORTS_DIR / "retailrocket_model_training_summary.csv"
    output_path = PLOTS_DIR / "retailrocket_model_metrics.png"

    if not input_path.exists():
        print(f"Skipping model metrics plot. Missing: {input_path}")
        return

    df = pd.read_csv(input_path)

    metric_columns = [
        "hit_rate_at_10",
        "precision_at_10",
        "recall_at_10",
        "ndcg_at_10",
    ]

    available_metrics = [column for column in metric_columns if column in df.columns]

    if not available_metrics:
        print("Skipping model metrics plot. No expected metrics found.")
        return

    values = [float(df.iloc[0][column]) for column in available_metrics]

    plt.figure(figsize=(8, 5))
    plt.bar(available_metrics, values)
    plt.title("Retailrocket Popularity Recommender Metrics")
    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    print(f"Saved plot: {output_path}")


def generate_sql_schema_summary() -> None:
    if not WAREHOUSE_PATH.exists():
        print(f"Skipping SQL schema summary. Missing: {WAREHOUSE_PATH}")
        return

    query = """
        SELECT
            table_schema AS schema_name,
            table_name,
            column_name,
            ordinal_position,
            data_type,
            is_nullable
        FROM information_schema.columns
        WHERE table_schema IN ('staged', 'mart', 'features')
        ORDER BY table_schema, table_name, ordinal_position;
    """

    with duckdb.connect(str(WAREHOUSE_PATH)) as connection:
        df = connection.execute(query).fetchdf()

    df.to_csv(SQL_SCHEMA_SUMMARY_PATH, index=False)
    print(f"Saved SQL schema summary: {SQL_SCHEMA_SUMMARY_PATH}")


def generate_feature_logic_summary() -> None:
    rows = [
        {
            "feature_group": "retailrocket_user_features",
            "feature_or_group": "total_events",
            "source": "mart.retailrocket_interactions",
            "logic": "Count all events generated by each user.",
            "used_for": "User activity representation, training, inference analysis",
        },
        {
            "feature_group": "retailrocket_user_features",
            "feature_or_group": "unique_items",
            "source": "mart.retailrocket_interactions",
            "logic": "Count distinct items interacted with by each user.",
            "used_for": "User diversity and activity profiling",
        },
        {
            "feature_group": "retailrocket_user_features",
            "feature_or_group": "view_events, addtocart_events, transaction_events",
            "source": "mart.retailrocket_interactions",
            "logic": "Conditional event counts by user and event type.",
            "used_for": "Intent-strength measurement",
        },
        {
            "feature_group": "retailrocket_user_features",
            "feature_or_group": "addtocart_event_rate, transaction_event_rate",
            "source": "features.retailrocket_user_features",
            "logic": "Event-type count divided by total user event count.",
            "used_for": "User conversion tendency",
        },
        {
            "feature_group": "retailrocket_item_features",
            "feature_or_group": "total_events, unique_users",
            "source": "mart.retailrocket_interactions",
            "logic": "Count total item interactions and distinct users per item.",
            "used_for": "Item popularity and candidate ranking",
        },
        {
            "feature_group": "retailrocket_item_features",
            "feature_or_group": "category_id, parent_category_id",
            "source": "staged.retailrocket_item_category_latest, staged.retailrocket_category_tree",
            "logic": "Attach latest item category and parent category metadata.",
            "used_for": "Content-aware filtering and catalog analysis",
        },
        {
            "feature_group": "retailrocket_item_features",
            "feature_or_group": "is_available",
            "source": "staged.retailrocket_item_availability_latest",
            "logic": "Attach latest availability flag for each item.",
            "used_for": "Candidate filtering and item metadata enrichment",
        },
        {
            "feature_group": "retailrocket_item_features",
            "feature_or_group": "addtocart_rate, transaction_rate, cart_to_transaction_rate",
            "source": "features.retailrocket_item_features",
            "logic": "Conversion ratios derived from item-level event counts.",
            "used_for": "Ranking and item quality signals",
        },
        {
            "feature_group": "retailrocket_user_item_features",
            "feature_or_group": "view_count, addtocart_count, transaction_count",
            "source": "mart.retailrocket_user_item_interactions",
            "logic": "Aggregate user-item event counts by event type.",
            "used_for": "Implicit feedback model training",
        },
        {
            "feature_group": "retailrocket_user_item_features",
            "feature_or_group": "implicit_label",
            "source": "features.retailrocket_user_item_features",
            "logic": "Encode interaction strength: view < addtocart < transaction.",
            "used_for": "Recommendation target signal",
        },
        {
            "feature_group": "retailrocket_user_item_features",
            "feature_or_group": "strongest_event_type",
            "source": "features.retailrocket_user_item_features",
            "logic": "Select strongest observed user-item event type.",
            "used_for": "Interpretability and behavioral analysis",
        },
        {
            "feature_group": "dummyjson_feature_registry",
            "feature_or_group": "user, item, interaction feature views",
            "source": "configs/feature_store/dummyjson_feature_registry.json",
            "logic": "Custom registry maps feature views to entities, source paths, and intended usage.",
            "used_for": "Feature store demonstration",
        },
    ]

    df = pd.DataFrame(rows)
    df.to_csv(FEATURE_LOGIC_SUMMARY_PATH, index=False)
    print(f"Saved feature logic summary: {FEATURE_LOGIC_SUMMARY_PATH}")


def classify_feature_column(column_name: str) -> str:
    entity_columns = {"user_id", "item_id", "cart_id"}
    metadata_columns = {
        "category_id",
        "parent_category_id",
        "is_available",
        "available_raw_value",
        "has_category_metadata",
        "has_availability_metadata",
        "_source_system",
        "_source_file",
        "_prepared_at",
    }

    if column_name in entity_columns:
        return "entity_key"

    if column_name in metadata_columns:
        return "metadata"

    return "feature"


def read_parquet_schema(path: Path) -> list[tuple[str, str]]:
    schema = pq.read_schema(path)
    return [(field.name, str(field.type)) for field in schema]


def generate_feature_metadata_documentation() -> None:
    feature_files = [
        {
            "feature_view_name": "retailrocket_user_features",
            "entity": "user_id",
            "source_path": "data/features/source=retailrocket/user_features.parquet",
            "used_for": "training, batch inference, user profiling",
        },
        {
            "feature_view_name": "retailrocket_item_features",
            "entity": "item_id",
            "source_path": "data/features/source=retailrocket/item_features.parquet",
            "used_for": "training, candidate ranking, item profiling",
        },
        {
            "feature_view_name": "retailrocket_user_item_features",
            "entity": "user_id,item_id",
            "source_path": "data/features/source=retailrocket/user_item_features.parquet",
            "used_for": "training, implicit feedback modeling",
        },
        {
            "feature_view_name": "dummyjson_user_features",
            "entity": "user_id",
            "source_path": "data/features/source=dummyjson/user_features.parquet",
            "used_for": "training, batch inference",
        },
        {
            "feature_view_name": "dummyjson_item_features",
            "entity": "item_id",
            "source_path": "data/features/source=dummyjson/item_features.parquet",
            "used_for": "training, batch inference, candidate filtering",
        },
        {
            "feature_view_name": "dummyjson_interaction_features",
            "entity": "user_id,item_id,cart_id",
            "source_path": "data/features/source=dummyjson/interaction_features.parquet",
            "used_for": "training",
        },
    ]

    rows = []

    for feature_file in feature_files:
        source_path = Path(feature_file["source_path"])

        if not source_path.exists():
            rows.append(
                {
                    **feature_file,
                    "column_name": None,
                    "data_type": None,
                    "column_role": "missing_source_file",
                    "source_exists": False,
                    "version": "v1",
                }
            )
            continue

        for column_name, data_type in read_parquet_schema(source_path):
            rows.append(
                {
                    **feature_file,
                    "column_name": column_name,
                    "data_type": data_type,
                    "column_role": classify_feature_column(column_name),
                    "source_exists": True,
                    "version": "v1",
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(FEATURE_METADATA_DOCUMENTATION_PATH, index=False)
    print(f"Saved feature metadata documentation: {FEATURE_METADATA_DOCUMENTATION_PATH}")


def parse_dvc_file(path: Path) -> dict[str, str | int | None]:
    text = path.read_text()

    def find_value(pattern: str) -> str | None:
        match = re.search(pattern, text)
        return match.group(1).strip() if match else None

    size_raw = find_value(r"size:\s*([0-9]+)")
    nfiles_raw = find_value(r"nfiles:\s*([0-9]+)")

    return {
        "dvc_file": str(path),
        "tracked_path": find_value(r"path:\s*(.+)"),
        "md5": find_value(r"md5:\s*(.+)"),
        "size_bytes": int(size_raw) if size_raw else None,
        "nfiles": int(nfiles_raw) if nfiles_raw else None,
    }


def generate_dvc_versioning_summary() -> None:
    dvc_files = [
        {
            "pipeline_layer": "external_data",
            "description": "Retailrocket original external CSV dataset",
            "file": Path("data/external/retailrocket.dvc"),
        },
        {
            "pipeline_layer": "staged_data",
            "description": "DummyJSON staged Parquet data",
            "file": Path("data/staged/source=dummyjson.dvc"),
        },
        {
            "pipeline_layer": "feature_data",
            "description": "DummyJSON feature Parquet data",
            "file": Path("data/features/source=dummyjson.dvc"),
        },
        {
            "pipeline_layer": "model_artifact",
            "description": "DummyJSON trained popularity model",
            "file": Path("models/dummyjson.dvc"),
        },
        {
            "pipeline_layer": "staged_data",
            "description": "Retailrocket staged Parquet data",
            "file": Path("data/staged/source=retailrocket.dvc"),
        },
        {
            "pipeline_layer": "feature_data",
            "description": "Retailrocket feature Parquet data",
            "file": Path("data/features/source=retailrocket.dvc"),
        },
        {
            "pipeline_layer": "model_artifact",
            "description": "Retailrocket trained popularity model",
            "file": Path("models/retailrocket.dvc"),
        },
    ]

    rows = []

    for item in dvc_files:
        dvc_file = item["file"]

        if dvc_file.exists():
            parsed = parse_dvc_file(dvc_file)
            rows.append(
                {
                    "pipeline_layer": item["pipeline_layer"],
                    "description": item["description"],
                    "dvc_file": parsed["dvc_file"],
                    "tracked_path": parsed["tracked_path"],
                    "md5": parsed["md5"],
                    "size_bytes": parsed["size_bytes"],
                    "nfiles": parsed["nfiles"],
                    "versioning_tool": "DVC",
                    "tracked_in_git": True,
                }
            )
        else:
            rows.append(
                {
                    "pipeline_layer": item["pipeline_layer"],
                    "description": item["description"],
                    "dvc_file": str(dvc_file),
                    "tracked_path": None,
                    "md5": None,
                    "size_bytes": None,
                    "nfiles": None,
                    "versioning_tool": "DVC",
                    "tracked_in_git": False,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(DVC_VERSIONING_SUMMARY_PATH, index=False)
    print(f"Saved DVC versioning summary: {DVC_VERSIONING_SUMMARY_PATH}")


def generate_repository_structure_summary() -> None:
    rows = [
        {
            "path": "configs/feature_store/",
            "purpose": "Custom feature registry configuration",
            "versioning": "Git",
            "example_files": "dummyjson_feature_registry.json",
        },
        {
            "path": "data/external/",
            "purpose": "External source datasets",
            "versioning": "DVC metadata in Git",
            "example_files": "retailrocket.dvc",
        },
        {
            "path": "data/raw/",
            "purpose": "Raw API ingested data",
            "versioning": "Ignored by Git; reproducible through ingestion",
            "example_files": "source=dummyjson/type=*/ingest_date=*",
        },
        {
            "path": "data/staged/",
            "purpose": "Cleaned staged Parquet datasets",
            "versioning": "DVC metadata in Git",
            "example_files": "source=dummyjson.dvc, source=retailrocket.dvc",
        },
        {
            "path": "data/features/",
            "purpose": "ML feature datasets",
            "versioning": "DVC metadata in Git",
            "example_files": "source=dummyjson.dvc, source=retailrocket.dvc",
        },
        {
            "path": "data/warehouse/",
            "purpose": "Local DuckDB analytical warehouse",
            "versioning": "Generated artifact, ignored by Git",
            "example_files": "recomart.duckdb",
        },
        {
            "path": "models/",
            "purpose": "Trained model artifacts",
            "versioning": "DVC metadata in Git",
            "example_files": "dummyjson.dvc, retailrocket.dvc",
        },
        {
            "path": "reports/",
            "purpose": "Reports, plots, screenshots, summaries",
            "versioning": "Git for lightweight files",
            "example_files": "final_project_report.md, plots/, screenshots/",
        },
        {
            "path": "sql/",
            "purpose": "Warehouse and feature transformation SQL",
            "versioning": "Git",
            "example_files": "retailrocket_warehouse.sql, retailrocket_features.sql",
        },
        {
            "path": "src/",
            "purpose": "Pipeline source code organized by stage",
            "versioning": "Git",
            "example_files": "ingestion/, validation/, preparation/, transformation/, training/, serving/",
        },
        {
            "path": "orchestration/",
            "purpose": "Prefect pipeline definitions",
            "versioning": "Git",
            "example_files": "dummyjson_pipeline.py, retailrocket_pipeline.py",
        },
    ]

    df = pd.DataFrame(rows)
    df.to_csv(REPOSITORY_STRUCTURE_SUMMARY_PATH, index=False)
    print(f"Saved repository structure summary: {REPOSITORY_STRUCTURE_SUMMARY_PATH}")


def main() -> None:
    ensure_dirs()

    save_event_distribution_plot()
    save_top_items_plot()
    save_user_activity_distribution_plot()
    save_model_metrics_plot()

    generate_sql_schema_summary()
    generate_feature_logic_summary()
    generate_feature_metadata_documentation()
    generate_dvc_versioning_summary()
    generate_repository_structure_summary()

    print("\nAssignment evidence artifact generation completed.")


if __name__ == "__main__":
    main()