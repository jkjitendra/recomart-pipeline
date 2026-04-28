from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGED_DIR = PROJECT_ROOT / "data/staged/source=retailrocket"
CURATED_DIR = PROJECT_ROOT / "data/curated/source=retailrocket"

EVENTS_PARQUET = STAGED_DIR / "events.parquet"
CATEGORY_TREE_PARQUET = STAGED_DIR / "category_tree.parquet"
ITEM_PROPERTIES_BATCH_PARQUET = STAGED_DIR / "item_properties_batch.parquet"
ITEM_PROPERTIES_API_DELTA_PARQUET = STAGED_DIR / "item_properties_api_delta.parquet"
ITEM_PROPERTIES_COMBINED_PARQUET = STAGED_DIR / "item_properties_combined.parquet"
ITEM_CATEGORY_LATEST_PARQUET = STAGED_DIR / "item_category_latest.parquet"
ITEM_AVAILABILITY_LATEST_PARQUET = STAGED_DIR / "item_availability_latest.parquet"

CURATED_INTERACTIONS_PARQUET = CURATED_DIR / "curated_interactions.parquet"
CURATED_ITEMS_PARQUET = CURATED_DIR / "curated_items.parquet"
CURATED_USER_ITEM_INTERACTIONS_PARQUET = CURATED_DIR / "curated_user_item_interactions.parquet"

REPORT_PATH = PROJECT_ROOT / "reports/data_quality/retailrocket_staged_validation_report.csv"


EXPECTED_FILES = {
    "events": EVENTS_PARQUET,
    "category_tree": CATEGORY_TREE_PARQUET,
    "item_properties_batch": ITEM_PROPERTIES_BATCH_PARQUET,
    "item_properties_api_delta": ITEM_PROPERTIES_API_DELTA_PARQUET,
    "item_properties_combined": ITEM_PROPERTIES_COMBINED_PARQUET,
    "item_category_latest": ITEM_CATEGORY_LATEST_PARQUET,
    "item_availability_latest": ITEM_AVAILABILITY_LATEST_PARQUET,
    "curated_interactions": CURATED_INTERACTIONS_PARQUET,
    "curated_items": CURATED_ITEMS_PARQUET,
    "curated_user_item_interactions": CURATED_USER_ITEM_INTERACTIONS_PARQUET,
}


REQUIRED_COLUMNS = {
    "events": [
        "event_timestamp_ms",
        "event_datetime",
        "visitor_id",
        "event_type",
        "item_id",
        "transaction_id",
        "_source_system",
        "_source_file",
    ],
    "category_tree": [
        "category_id",
        "parent_category_id",
        "is_root_category",
        "_source_system",
        "_source_file",
    ],
    "item_properties_batch": [
        "property_timestamp_ms",
        "property_datetime",
        "item_id",
        "property_name",
        "property_value",
        "source_feed",
        "source_system",
        "source_file",
    ],
    "item_properties_api_delta": [
        "property_timestamp_ms",
        "property_datetime",
        "item_id",
        "property_name",
        "property_value",
        "source_feed",
        "source_system",
        "source_file",
        "ingestion_timestamp",
    ],
    "item_properties_combined": [
        "property_timestamp_ms",
        "property_datetime",
        "item_id",
        "property_name",
        "property_value",
        "source_feed",
        "source_system",
        "source_file",
        "ingestion_timestamp",
    ],
    "item_category_latest": [
        "item_id",
        "category_id",
        "category_timestamp_ms",
        "category_datetime",
        "_source_system",
        "_source_file",
    ],
    "item_availability_latest": [
        "item_id",
        "available_raw_value",
        "is_available",
        "availability_timestamp_ms",
        "availability_datetime",
        "_source_system",
        "_source_file",
    ],
    "curated_interactions": [
        "user_id",
        "item_id",
        "event_type",
        "event_timestamp_ms",
        "interaction_weight",
        "category_id",
        "is_available",
    ],
    "curated_items": [
        "item_id",
        "total_events",
        "unique_users",
        "category_id",
        "is_available",
        "has_category_metadata",
        "has_availability_metadata",
    ],
    "curated_user_item_interactions": [
        "user_id",
        "item_id",
        "total_events",
        "view_count",
        "addtocart_count",
        "transaction_count",
        "interaction_score",
        "has_transaction",
        "has_addtocart",
    ],
}


def display_path(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)


def add_result(
    rows: list[dict[str, Any]],
    dataset_name: str,
    check_name: str,
    status: str,
    details: str,
) -> None:
    rows.append(
        {
            "dataset_name": dataset_name,
            "check_name": check_name,
            "status": status,
            "details": details,
        }
    )


def parquet_relation(path: Path) -> str:
    return f"read_parquet('{path}', hive_partitioning = false)"


def get_columns(connection: duckdb.DuckDBPyConnection, path: Path) -> list[str]:
    return list(connection.execute(f"SELECT * FROM {parquet_relation(path)} LIMIT 0").fetchdf().columns)


def get_row_count(connection: duckdb.DuckDBPyConnection, path: Path) -> int:
    return int(connection.execute(f"SELECT COUNT(*) FROM {parquet_relation(path)}").fetchone()[0])


def validate_files_and_schema(connection: duckdb.DuckDBPyConnection, rows: list[dict[str, Any]]) -> None:
    for dataset_name, path in EXPECTED_FILES.items():
        if not path.exists():
            add_result(rows, dataset_name, "file_exists", "FAIL", f"Missing file: {display_path(path)}")
            continue

        add_result(rows, dataset_name, "file_exists", "PASS", f"Found file: {display_path(path)}")

        try:
            row_count = get_row_count(connection, path)
            columns = get_columns(connection, path)

            add_result(rows, dataset_name, "read_parquet", "PASS", f"{row_count} rows, {len(columns)} columns")

            missing_columns = sorted(set(REQUIRED_COLUMNS[dataset_name]) - set(columns))
            if missing_columns:
                add_result(rows, dataset_name, "required_columns", "FAIL", f"Missing columns: {missing_columns}")
            else:
                add_result(rows, dataset_name, "required_columns", "PASS", "All required columns found")

            add_result(
                rows,
                dataset_name,
                "row_count_positive",
                "PASS" if row_count > 0 else "FAIL",
                f"Found {row_count} rows",
            )
        except Exception as exc:
            add_result(rows, dataset_name, "read_parquet", "FAIL", str(exc))


def validate_events(connection: duckdb.DuckDBPyConnection, rows: list[dict[str, Any]]) -> None:
    relation = parquet_relation(EVENTS_PARQUET)

    invalid_event_types = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {relation}
        WHERE event_type NOT IN ('view', 'addtocart', 'transaction')
        """
    ).fetchone()[0]
    add_result(
        rows,
        "events",
        "valid_event_types",
        "PASS" if invalid_event_types == 0 else "FAIL",
        f"Invalid rows: {invalid_event_types}",
    )

    invalid_timestamps = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {relation}
        WHERE event_timestamp_ms IS NULL
           OR event_timestamp_ms <= 0
           OR event_datetime IS NULL
        """
    ).fetchone()[0]
    add_result(
        rows,
        "events",
        "valid_timestamps",
        "PASS" if invalid_timestamps == 0 else "FAIL",
        f"Invalid timestamp rows: {invalid_timestamps}",
    )

    null_counts = connection.execute(
        f"""
        SELECT
            SUM(CASE WHEN visitor_id IS NULL THEN 1 ELSE 0 END) AS null_visitor_id,
            SUM(CASE WHEN event_type IS NULL THEN 1 ELSE 0 END) AS null_event_type,
            SUM(CASE WHEN item_id IS NULL THEN 1 ELSE 0 END) AS null_item_id
        FROM {relation}
        """
    ).fetchdf().iloc[0].to_dict()

    for column_name, null_count in null_counts.items():
        add_result(
            rows,
            "events",
            column_name,
            "PASS" if int(null_count) == 0 else "FAIL",
            f"Null rows: {int(null_count)}",
        )

    duplicate_rows = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT event_timestamp_ms, visitor_id, event_type, item_id, transaction_id, COUNT(*) AS row_count
            FROM {relation}
            GROUP BY event_timestamp_ms, visitor_id, event_type, item_id, transaction_id
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]
    add_result(
        rows,
        "events",
        "duplicate_event_rows",
        "PASS" if duplicate_rows == 0 else "WARN",
        f"Duplicate event groups: {duplicate_rows}",
    )


def validate_item_properties(connection: duckdb.DuckDBPyConnection, rows: list[dict[str, Any]]) -> None:
    batch_relation = parquet_relation(ITEM_PROPERTIES_BATCH_PARQUET)
    api_relation = parquet_relation(ITEM_PROPERTIES_API_DELTA_PARQUET)
    combined_relation = parquet_relation(ITEM_PROPERTIES_COMBINED_PARQUET)
    category_relation = parquet_relation(ITEM_CATEGORY_LATEST_PARQUET)
    availability_relation = parquet_relation(ITEM_AVAILABILITY_LATEST_PARQUET)

    invalid_batch_properties = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {batch_relation}
        WHERE property_name NOT IN ('categoryid', 'available')
        """
    ).fetchone()[0]
    add_result(
        rows,
        "item_properties_batch",
        "selected_property_names",
        "PASS" if invalid_batch_properties == 0 else "FAIL",
        f"Unexpected batch property rows: {invalid_batch_properties}",
    )

    api_nulls = connection.execute(
        f"""
        SELECT
            SUM(CASE WHEN property_timestamp_ms IS NULL THEN 1 ELSE 0 END) AS null_property_timestamp_ms,
            SUM(CASE WHEN item_id IS NULL THEN 1 ELSE 0 END) AS null_item_id,
            SUM(CASE WHEN property_name IS NULL THEN 1 ELSE 0 END) AS null_property_name,
            SUM(CASE WHEN property_value IS NULL THEN 1 ELSE 0 END) AS null_property_value
        FROM {api_relation}
        """
    ).fetchdf().iloc[0].to_dict()

    for column_name, null_count in api_nulls.items():
        add_result(
            rows,
            "item_properties_api_delta",
            column_name,
            "PASS" if int(null_count) == 0 else "FAIL",
            f"Null rows: {int(null_count)}",
        )

    batch_count, api_count, combined_count = connection.execute(
        f"""
        SELECT
            (SELECT COUNT(*) FROM {batch_relation}) AS batch_count,
            (SELECT COUNT(*) FROM {api_relation}) AS api_count,
            (SELECT COUNT(*) FROM {combined_relation}) AS combined_count
        """
    ).fetchone()
    add_result(
        rows,
        "item_properties_combined",
        "combined_row_count",
        "PASS" if int(batch_count) + int(api_count) == int(combined_count) else "FAIL",
        f"batch={batch_count}, api={api_count}, combined={combined_count}",
    )

    invalid_property_timestamps = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {combined_relation}
        WHERE property_timestamp_ms IS NULL
           OR property_timestamp_ms <= 0
           OR property_datetime IS NULL
        """
    ).fetchone()[0]
    add_result(
        rows,
        "item_properties_combined",
        "valid_property_timestamps",
        "PASS" if invalid_property_timestamps == 0 else "FAIL",
        f"Invalid rows: {invalid_property_timestamps}",
    )

    for dataset_name, relation in [
        ("item_category_latest", category_relation),
        ("item_availability_latest", availability_relation),
    ]:
        duplicate_items = connection.execute(
            f"""
            SELECT COUNT(*)
            FROM (
                SELECT item_id
                FROM {relation}
                GROUP BY item_id
                HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]
        add_result(
            rows,
            dataset_name,
            "unique_item_id",
            "PASS" if duplicate_items == 0 else "FAIL",
            f"Duplicate item IDs: {duplicate_items}",
        )

    invalid_availability = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {availability_relation}
        WHERE is_available NOT IN (0, 1)
           OR is_available IS NULL
        """
    ).fetchone()[0]
    add_result(
        rows,
        "item_availability_latest",
        "valid_availability_values",
        "PASS" if invalid_availability == 0 else "FAIL",
        f"Invalid availability rows: {invalid_availability}",
    )


def validate_category_tree(connection: duckdb.DuckDBPyConnection, rows: list[dict[str, Any]]) -> None:
    relation = parquet_relation(CATEGORY_TREE_PARQUET)

    duplicate_categories = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT category_id
            FROM {relation}
            GROUP BY category_id
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]
    add_result(
        rows,
        "category_tree",
        "unique_category_id",
        "PASS" if duplicate_categories == 0 else "FAIL",
        f"Duplicate category IDs: {duplicate_categories}",
    )


def validate_referential_coverage(connection: duckdb.DuckDBPyConnection, rows: list[dict[str, Any]]) -> None:
    events_relation = parquet_relation(EVENTS_PARQUET)
    category_relation = parquet_relation(ITEM_CATEGORY_LATEST_PARQUET)
    availability_relation = parquet_relation(ITEM_AVAILABILITY_LATEST_PARQUET)

    coverage = connection.execute(
        f"""
        WITH event_items AS (
            SELECT DISTINCT item_id
            FROM {events_relation}
        )
        SELECT
            COUNT(*) AS event_distinct_items,
            SUM(CASE WHEN c.item_id IS NOT NULL THEN 1 ELSE 0 END) AS items_with_category,
            SUM(CASE WHEN a.item_id IS NOT NULL THEN 1 ELSE 0 END) AS items_with_availability
        FROM event_items e
        LEFT JOIN {category_relation} c
            ON e.item_id = c.item_id
        LEFT JOIN {availability_relation} a
            ON e.item_id = a.item_id
        """
    ).fetchdf().iloc[0].to_dict()

    event_distinct_items = int(coverage["event_distinct_items"])
    items_with_category = int(coverage["items_with_category"])
    items_with_availability = int(coverage["items_with_availability"])

    category_coverage = items_with_category / event_distinct_items if event_distinct_items else 0
    availability_coverage = items_with_availability / event_distinct_items if event_distinct_items else 0

    add_result(
        rows,
        "referential_coverage",
        "event_items_with_category",
        "PASS" if category_coverage >= 0.5 else "WARN",
        f"{items_with_category}/{event_distinct_items} ({category_coverage:.4f})",
    )
    add_result(
        rows,
        "referential_coverage",
        "event_items_with_availability",
        "PASS" if availability_coverage >= 0.5 else "WARN",
        f"{items_with_availability}/{event_distinct_items} ({availability_coverage:.4f})",
    )


def validate_curated_outputs(connection: duckdb.DuckDBPyConnection, rows: list[dict[str, Any]]) -> None:
    curated_interactions = parquet_relation(CURATED_INTERACTIONS_PARQUET)
    curated_items = parquet_relation(CURATED_ITEMS_PARQUET)
    curated_user_items = parquet_relation(CURATED_USER_ITEM_INTERACTIONS_PARQUET)

    invalid_weights = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {curated_interactions}
        WHERE interaction_weight NOT IN (1.0, 3.0, 5.0)
        """
    ).fetchone()[0]
    add_result(
        rows,
        "curated_interactions",
        "valid_interaction_weights",
        "PASS" if invalid_weights == 0 else "FAIL",
        f"Invalid weights: {invalid_weights}",
    )

    duplicate_items = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT item_id
            FROM {curated_items}
            GROUP BY item_id
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]
    add_result(
        rows,
        "curated_items",
        "unique_item_id",
        "PASS" if duplicate_items == 0 else "FAIL",
        f"Duplicate item IDs: {duplicate_items}",
    )

    duplicate_user_items = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT user_id, item_id
            FROM {curated_user_items}
            GROUP BY user_id, item_id
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]
    add_result(
        rows,
        "curated_user_item_interactions",
        "unique_user_item_key",
        "PASS" if duplicate_user_items == 0 else "FAIL",
        f"Duplicate user-item keys: {duplicate_user_items}",
    )


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    with duckdb.connect(database=":memory:") as connection:
        validate_files_and_schema(connection, rows)
        validate_events(connection, rows)
        validate_category_tree(connection, rows)
        validate_item_properties(connection, rows)
        validate_referential_coverage(connection, rows)
        validate_curated_outputs(connection, rows)

    report_df = pd.DataFrame(rows)
    report_df.to_csv(REPORT_PATH, index=False)

    total_checks = len(report_df)
    passed_checks = int((report_df["status"] == "PASS").sum())
    warning_checks = int((report_df["status"] == "WARN").sum())
    failed_checks = int((report_df["status"] == "FAIL").sum())

    print(f"Retailrocket staged validation report saved: {display_path(REPORT_PATH)}")
    print(f"Total checks: {total_checks}")
    print(f"Passed checks: {passed_checks}")
    print(f"Warnings: {warning_checks}")
    print(f"Failed checks: {failed_checks}")

    if warning_checks > 0:
        print("\nWarning checks:")
        print(report_df[report_df["status"] == "WARN"].to_string(index=False))

    if failed_checks > 0:
        print("\nFailed checks:")
        print(report_df[report_df["status"] == "FAIL"].to_string(index=False))
        raise RuntimeError(f"Retailrocket staged validation failed with {failed_checks} failed checks.")

    print("\nNo failed Retailrocket staged validation checks.")


if __name__ == "__main__":
    main()
