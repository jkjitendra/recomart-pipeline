from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


EXTERNAL_DIR = Path("data/external/retailrocket")

EVENTS_PATH = EXTERNAL_DIR / "events.csv"
ITEM_PROPERTIES_PART1_PATH = EXTERNAL_DIR / "item_properties_part1.csv"
ITEM_PROPERTIES_PART2_PATH = EXTERNAL_DIR / "item_properties_part2.csv"
CATEGORY_TREE_PATH = EXTERNAL_DIR / "category_tree.csv"

SUMMARY_REPORT_PATH = Path("reports/retailrocket_external_summary.csv")
EVENT_TYPE_REPORT_PATH = Path("reports/retailrocket_event_type_summary.csv")
TOP_PROPERTIES_REPORT_PATH = Path("reports/retailrocket_top_item_properties.csv")
CATEGORY_TREE_REPORT_PATH = Path("reports/retailrocket_category_tree_summary.csv")


def ensure_files_exist() -> None:
    required_files = [
        EVENTS_PATH,
        ITEM_PROPERTIES_PART1_PATH,
        ITEM_PROPERTIES_PART2_PATH,
        CATEGORY_TREE_PATH,
    ]

    missing_files = [str(file_path) for file_path in required_files if not file_path.exists()]

    if missing_files:
        raise FileNotFoundError(f"Missing Retailrocket files: {missing_files}")


def milliseconds_to_datetime(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None

    return pd.to_datetime(int(value), unit="ms").isoformat()


def file_size_mb(file_path: Path) -> float:
    return round(file_path.stat().st_size / (1024 * 1024), 3)


def read_single_value(connection: duckdb.DuckDBPyConnection, query: str) -> Any:
    return connection.execute(query).fetchone()[0]


def inspect_events(connection: duckdb.DuckDBPyConnection) -> tuple[dict[str, Any], pd.DataFrame]:
    events_relation = f"read_csv_auto('{EVENTS_PATH}', header=true)"

    summary_query = f"""
        SELECT
            COUNT(*) AS row_count,
            COUNT(DISTINCT visitorid) AS distinct_visitors,
            COUNT(DISTINCT itemid) AS distinct_items,
            MIN(timestamp) AS min_timestamp_ms,
            MAX(timestamp) AS max_timestamp_ms,
            SUM(CASE WHEN transactionid IS NOT NULL THEN 1 ELSE 0 END) AS rows_with_transaction_id,
            SUM(CASE WHEN transactionid IS NULL THEN 1 ELSE 0 END) AS rows_without_transaction_id
        FROM {events_relation}
    """

    row = connection.execute(summary_query).fetchdf().iloc[0].to_dict()

    row["dataset_name"] = "events"
    row["file_path"] = str(EVENTS_PATH)
    row["file_size_mb"] = file_size_mb(EVENTS_PATH)
    row["min_timestamp_datetime"] = milliseconds_to_datetime(row["min_timestamp_ms"])
    row["max_timestamp_datetime"] = milliseconds_to_datetime(row["max_timestamp_ms"])

    event_type_query = f"""
        SELECT
            event,
            COUNT(*) AS row_count,
            COUNT(DISTINCT visitorid) AS distinct_visitors,
            COUNT(DISTINCT itemid) AS distinct_items,
            SUM(CASE WHEN transactionid IS NOT NULL THEN 1 ELSE 0 END) AS rows_with_transaction_id
        FROM {events_relation}
        GROUP BY event
        ORDER BY row_count DESC
    """

    event_type_df = connection.execute(event_type_query).fetchdf()

    return row, event_type_df


def inspect_item_properties(connection: duckdb.DuckDBPyConnection) -> tuple[dict[str, Any], pd.DataFrame]:
    part1_relation = f"read_csv_auto('{ITEM_PROPERTIES_PART1_PATH}', header=true)"
    part2_relation = f"read_csv_auto('{ITEM_PROPERTIES_PART2_PATH}', header=true)"

    combined_relation = f"""
        (
            SELECT * FROM {part1_relation}
            UNION ALL
            SELECT * FROM {part2_relation}
        )
    """

    summary_query = f"""
        SELECT
            COUNT(*) AS row_count,
            COUNT(DISTINCT itemid) AS distinct_items,
            COUNT(DISTINCT property) AS distinct_properties,
            MIN(timestamp) AS min_timestamp_ms,
            MAX(timestamp) AS max_timestamp_ms,
            SUM(CASE WHEN value IS NULL THEN 1 ELSE 0 END) AS null_value_rows
        FROM {combined_relation}
    """

    row = connection.execute(summary_query).fetchdf().iloc[0].to_dict()

    row["dataset_name"] = "item_properties_combined"
    row["file_path"] = f"{ITEM_PROPERTIES_PART1_PATH}; {ITEM_PROPERTIES_PART2_PATH}"
    row["file_size_mb"] = file_size_mb(ITEM_PROPERTIES_PART1_PATH) + file_size_mb(ITEM_PROPERTIES_PART2_PATH)
    row["min_timestamp_datetime"] = milliseconds_to_datetime(row["min_timestamp_ms"])
    row["max_timestamp_datetime"] = milliseconds_to_datetime(row["max_timestamp_ms"])

    top_properties_query = f"""
        SELECT
            property,
            COUNT(*) AS row_count,
            COUNT(DISTINCT itemid) AS distinct_items,
            COUNT(DISTINCT value) AS distinct_values
        FROM {combined_relation}
        GROUP BY property
        ORDER BY row_count DESC
        LIMIT 30
    """

    top_properties_df = connection.execute(top_properties_query).fetchdf()

    return row, top_properties_df


def inspect_category_tree(connection: duckdb.DuckDBPyConnection) -> tuple[dict[str, Any], pd.DataFrame]:
    category_relation = f"read_csv_auto('{CATEGORY_TREE_PATH}', header=true)"

    summary_query = f"""
        SELECT
            COUNT(*) AS row_count,
            COUNT(DISTINCT categoryid) AS distinct_categories,
            COUNT(DISTINCT parentid) AS distinct_parent_categories,
            SUM(CASE WHEN parentid IS NULL THEN 1 ELSE 0 END) AS root_category_rows
        FROM {category_relation}
    """

    row = connection.execute(summary_query).fetchdf().iloc[0].to_dict()

    row["dataset_name"] = "category_tree"
    row["file_path"] = str(CATEGORY_TREE_PATH)
    row["file_size_mb"] = file_size_mb(CATEGORY_TREE_PATH)

    category_tree_sample_query = f"""
        SELECT *
        FROM {category_relation}
        LIMIT 20
    """

    category_sample_df = connection.execute(category_tree_sample_query).fetchdf()

    return row, category_sample_df


def write_reports(
    summary_rows: list[dict[str, Any]],
    event_type_df: pd.DataFrame,
    top_properties_df: pd.DataFrame,
    category_tree_sample_df: pd.DataFrame,
) -> None:
    SUMMARY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(summary_rows)

    # Put common metadata columns first when present.
    preferred_columns = [
        "dataset_name",
        "file_path",
        "file_size_mb",
        "row_count",
        "distinct_visitors",
        "distinct_items",
        "distinct_properties",
        "distinct_categories",
        "distinct_parent_categories",
        "root_category_rows",
        "rows_with_transaction_id",
        "rows_without_transaction_id",
        "null_value_rows",
        "min_timestamp_ms",
        "max_timestamp_ms",
        "min_timestamp_datetime",
        "max_timestamp_datetime",
    ]

    ordered_columns = [col for col in preferred_columns if col in summary_df.columns]
    remaining_columns = [col for col in summary_df.columns if col not in ordered_columns]
    summary_df = summary_df[ordered_columns + remaining_columns]

    summary_df.to_csv(SUMMARY_REPORT_PATH, index=False)
    event_type_df.to_csv(EVENT_TYPE_REPORT_PATH, index=False)
    top_properties_df.to_csv(TOP_PROPERTIES_REPORT_PATH, index=False)
    category_tree_sample_df.to_csv(CATEGORY_TREE_REPORT_PATH, index=False)

    print(f"Summary report saved: {SUMMARY_REPORT_PATH}")
    print(f"Event type report saved: {EVENT_TYPE_REPORT_PATH}")
    print(f"Top item properties report saved: {TOP_PROPERTIES_REPORT_PATH}")
    print(f"Category tree sample report saved: {CATEGORY_TREE_REPORT_PATH}")


def main() -> None:
    ensure_files_exist()

    with duckdb.connect(database=":memory:") as connection:
        events_summary, event_type_df = inspect_events(connection)
        item_properties_summary, top_properties_df = inspect_item_properties(connection)
        category_tree_summary, category_tree_sample_df = inspect_category_tree(connection)

    summary_rows = [
        events_summary,
        item_properties_summary,
        category_tree_summary,
    ]

    write_reports(
        summary_rows=summary_rows,
        event_type_df=event_type_df,
        top_properties_df=top_properties_df,
        category_tree_sample_df=category_tree_sample_df,
    )

    print("\nRetailrocket external inspection completed.")
    print("\nDataset summary:")
    print(pd.DataFrame(summary_rows).to_string(index=False))

    print("\nEvent type summary:")
    print(event_type_df.to_string(index=False))

    print("\nTop item properties:")
    print(top_properties_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()