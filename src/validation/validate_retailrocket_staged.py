from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


STAGED_DIR = Path("data/staged/source=retailrocket")

EVENTS_PARQUET = STAGED_DIR / "events.parquet"
CATEGORY_TREE_PARQUET = STAGED_DIR / "category_tree.parquet"
ITEM_PROPERTIES_SELECTED_PARQUET = STAGED_DIR / "item_properties_selected.parquet"
ITEM_CATEGORY_LATEST_PARQUET = STAGED_DIR / "item_category_latest.parquet"
ITEM_AVAILABILITY_LATEST_PARQUET = STAGED_DIR / "item_availability_latest.parquet"

REPORT_PATH = Path("reports/data_quality/retailrocket_staged_validation_report.csv")


EXPECTED_FILES = {
    "events": EVENTS_PARQUET,
    "category_tree": CATEGORY_TREE_PARQUET,
    "item_properties_selected": ITEM_PROPERTIES_SELECTED_PARQUET,
    "item_category_latest": ITEM_CATEGORY_LATEST_PARQUET,
    "item_availability_latest": ITEM_AVAILABILITY_LATEST_PARQUET,
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
    "item_properties_selected": [
        "property_timestamp_ms",
        "property_datetime",
        "item_id",
        "property_name",
        "property_value",
        "_source_part",
        "_source_system",
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
}


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
    query = f"SELECT * FROM {parquet_relation(path)} LIMIT 0"
    return list(connection.execute(query).fetchdf().columns)


def get_row_count(connection: duckdb.DuckDBPyConnection, path: Path) -> int:
    query = f"SELECT COUNT(*) FROM {parquet_relation(path)}"
    return int(connection.execute(query).fetchone()[0])


def validate_files_and_schema(
    connection: duckdb.DuckDBPyConnection,
    rows: list[dict[str, Any]],
) -> None:
    for dataset_name, path in EXPECTED_FILES.items():
        if not path.exists():
            add_result(rows, dataset_name, "file_exists", "FAIL", f"Missing file: {path}")
            continue

        add_result(rows, dataset_name, "file_exists", "PASS", f"Found file: {path}")

        try:
            row_count = get_row_count(connection, path)
            columns = get_columns(connection, path)

            add_result(
                rows,
                dataset_name,
                "read_parquet",
                "PASS",
                f"Successfully read file with {row_count} rows and {len(columns)} columns",
            )

            missing_columns = sorted(set(REQUIRED_COLUMNS[dataset_name]) - set(columns))

            if missing_columns:
                add_result(
                    rows,
                    dataset_name,
                    "required_columns",
                    "FAIL",
                    f"Missing columns: {missing_columns}",
                )
            else:
                add_result(
                    rows,
                    dataset_name,
                    "required_columns",
                    "PASS",
                    f"All required columns found: {REQUIRED_COLUMNS[dataset_name]}",
                )

            if row_count > 0:
                add_result(rows, dataset_name, "row_count_positive", "PASS", f"Found {row_count} rows")
            else:
                add_result(rows, dataset_name, "row_count_positive", "FAIL", "Found 0 rows")

        except Exception as exc:
            add_result(rows, dataset_name, "read_parquet", "FAIL", str(exc))


def validate_events(connection: duckdb.DuckDBPyConnection, rows: list[dict[str, Any]]) -> None:
    relation = parquet_relation(EVENTS_PARQUET)

    valid_event_types = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {relation}
        WHERE event_type NOT IN ('view', 'addtocart', 'transaction')
        """
    ).fetchone()[0]

    if valid_event_types == 0:
        add_result(rows, "events", "valid_event_types", "PASS", "Only expected event types found")
    else:
        add_result(rows, "events", "valid_event_types", "FAIL", f"Found {valid_event_types} invalid event rows")

    null_counts = connection.execute(
        f"""
        SELECT
            SUM(CASE WHEN event_timestamp_ms IS NULL THEN 1 ELSE 0 END) AS null_timestamp,
            SUM(CASE WHEN visitor_id IS NULL THEN 1 ELSE 0 END) AS null_visitor,
            SUM(CASE WHEN event_type IS NULL THEN 1 ELSE 0 END) AS null_event_type,
            SUM(CASE WHEN item_id IS NULL THEN 1 ELSE 0 END) AS null_item
        FROM {relation}
        """
    ).fetchdf().iloc[0].to_dict()

    for column_name, null_count in null_counts.items():
        if int(null_count) == 0:
            add_result(rows, "events", f"{column_name}", "PASS", "No nulls found")
        else:
            add_result(rows, "events", f"{column_name}", "FAIL", f"Found {int(null_count)} nulls")

    transaction_logic = connection.execute(
        f"""
        SELECT
            SUM(CASE WHEN event_type = 'transaction' AND transaction_id IS NULL THEN 1 ELSE 0 END) AS transaction_missing_id,
            SUM(CASE WHEN event_type <> 'transaction' AND transaction_id IS NOT NULL THEN 1 ELSE 0 END) AS non_transaction_with_id
        FROM {relation}
        """
    ).fetchdf().iloc[0].to_dict()

    if int(transaction_logic["transaction_missing_id"]) == 0:
        add_result(
            rows,
            "events",
            "transaction_rows_have_transaction_id",
            "PASS",
            "All transaction rows have transaction_id",
        )
    else:
        add_result(
            rows,
            "events",
            "transaction_rows_have_transaction_id",
            "FAIL",
            f"{int(transaction_logic['transaction_missing_id'])} transaction rows missing transaction_id",
        )

    if int(transaction_logic["non_transaction_with_id"]) == 0:
        add_result(
            rows,
            "events",
            "non_transaction_rows_without_transaction_id",
            "PASS",
            "Non-transaction rows do not have transaction_id",
        )
    else:
        add_result(
            rows,
            "events",
            "non_transaction_rows_without_transaction_id",
            "WARN",
            f"{int(transaction_logic['non_transaction_with_id'])} non-transaction rows have transaction_id",
        )

    event_distribution = connection.execute(
        f"""
        SELECT event_type, COUNT(*) AS row_count
        FROM {relation}
        GROUP BY event_type
        ORDER BY row_count DESC
        """
    ).fetchdf()

    add_result(
        rows,
        "events",
        "event_type_distribution",
        "PASS",
        event_distribution.to_dict(orient="records").__str__(),
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

    if duplicate_categories == 0:
        add_result(rows, "category_tree", "unique_category_id", "PASS", "No duplicate category_id values")
    else:
        add_result(rows, "category_tree", "unique_category_id", "FAIL", f"Found {duplicate_categories} duplicate category IDs")

    invalid_roots = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {relation}
        WHERE parent_category_id IS NULL AND is_root_category <> 1
        """
    ).fetchone()[0]

    if invalid_roots == 0:
        add_result(rows, "category_tree", "root_category_flag", "PASS", "Root category flag is consistent")
    else:
        add_result(rows, "category_tree", "root_category_flag", "FAIL", f"Found {invalid_roots} inconsistent root flags")


def validate_item_properties(connection: duckdb.DuckDBPyConnection, rows: list[dict[str, Any]]) -> None:
    selected_relation = parquet_relation(ITEM_PROPERTIES_SELECTED_PARQUET)
    category_relation = parquet_relation(ITEM_CATEGORY_LATEST_PARQUET)
    availability_relation = parquet_relation(ITEM_AVAILABILITY_LATEST_PARQUET)

    invalid_property_names = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {selected_relation}
        WHERE property_name NOT IN ('categoryid', 'available')
        """
    ).fetchone()[0]

    if invalid_property_names == 0:
        add_result(
            rows,
            "item_properties_selected",
            "selected_property_names",
            "PASS",
            "Only categoryid and available are present",
        )
    else:
        add_result(
            rows,
            "item_properties_selected",
            "selected_property_names",
            "FAIL",
            f"Found {invalid_property_names} rows with unexpected properties",
        )

    latest_category_duplicates = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT item_id
            FROM {category_relation}
            GROUP BY item_id
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    if latest_category_duplicates == 0:
        add_result(
            rows,
            "item_category_latest",
            "unique_item_id",
            "PASS",
            "One latest category row per item",
        )
    else:
        add_result(
            rows,
            "item_category_latest",
            "unique_item_id",
            "FAIL",
            f"Found {latest_category_duplicates} duplicate item IDs",
        )

    latest_availability_duplicates = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT item_id
            FROM {availability_relation}
            GROUP BY item_id
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    if latest_availability_duplicates == 0:
        add_result(
            rows,
            "item_availability_latest",
            "unique_item_id",
            "PASS",
            "One latest availability row per item",
        )
    else:
        add_result(
            rows,
            "item_availability_latest",
            "unique_item_id",
            "FAIL",
            f"Found {latest_availability_duplicates} duplicate item IDs",
        )

    invalid_availability_values = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {availability_relation}
        WHERE is_available NOT IN (0, 1)
           OR is_available IS NULL
        """
    ).fetchone()[0]

    if invalid_availability_values == 0:
        add_result(
            rows,
            "item_availability_latest",
            "valid_availability_values",
            "PASS",
            "All availability values are 0 or 1",
        )
    else:
        add_result(
            rows,
            "item_availability_latest",
            "valid_availability_values",
            "FAIL",
            f"Found {invalid_availability_values} invalid availability values",
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
        f"{items_with_category}/{event_distinct_items} event items have latest category coverage ({category_coverage:.4f})",
    )

    add_result(
        rows,
        "referential_coverage",
        "event_items_with_availability",
        "PASS" if availability_coverage >= 0.5 else "WARN",
        f"{items_with_availability}/{event_distinct_items} event items have latest availability coverage ({availability_coverage:.4f})",
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

    report_df = pd.DataFrame(rows)
    report_df.to_csv(REPORT_PATH, index=False)

    total_checks = len(report_df)
    passed_checks = int((report_df["status"] == "PASS").sum())
    warning_checks = int((report_df["status"] == "WARN").sum())
    failed_checks = int((report_df["status"] == "FAIL").sum())

    print(f"Retailrocket staged validation report saved: {REPORT_PATH}")
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