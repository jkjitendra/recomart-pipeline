from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


EXTERNAL_DIR = Path("data/external/retailrocket")
STAGED_DIR = Path("data/staged/source=retailrocket")

EVENTS_CSV = EXTERNAL_DIR / "events.csv"
ITEM_PROPERTIES_PART1_CSV = EXTERNAL_DIR / "item_properties_part1.csv"
ITEM_PROPERTIES_PART2_CSV = EXTERNAL_DIR / "item_properties_part2.csv"
CATEGORY_TREE_CSV = EXTERNAL_DIR / "category_tree.csv"

EVENTS_PARQUET = STAGED_DIR / "events.parquet"
CATEGORY_TREE_PARQUET = STAGED_DIR / "category_tree.parquet"
ITEM_PROPERTIES_SELECTED_PARQUET = STAGED_DIR / "item_properties_selected.parquet"
ITEM_CATEGORY_LATEST_PARQUET = STAGED_DIR / "item_category_latest.parquet"
ITEM_AVAILABILITY_LATEST_PARQUET = STAGED_DIR / "item_availability_latest.parquet"

REPORT_PATH = Path("reports/retailrocket_preparation_summary.csv")


def ensure_input_files_exist() -> None:
    required_files = [
        EVENTS_CSV,
        ITEM_PROPERTIES_PART1_CSV,
        ITEM_PROPERTIES_PART2_CSV,
        CATEGORY_TREE_CSV,
    ]

    missing_files = [str(file_path) for file_path in required_files if not file_path.exists()]

    if missing_files:
        raise FileNotFoundError(f"Missing Retailrocket input files: {missing_files}")


def run_sql(connection: duckdb.DuckDBPyConnection, sql: str) -> None:
    connection.execute(sql)


def count_parquet_rows(connection: duckdb.DuckDBPyConnection, parquet_path: Path) -> int:
    query = f"SELECT COUNT(*) FROM read_parquet(\'{parquet_path}\', hive_partitioning = false)"
    return int(connection.execute(query).fetchone()[0])


def count_parquet_columns(connection: duckdb.DuckDBPyConnection, parquet_path: Path) -> int:
    query = f"SELECT * FROM read_parquet(\'{parquet_path}\', hive_partitioning = false) LIMIT 0"
    return len(connection.execute(query).fetchdf().columns)


def build_summary(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    outputs = [
        {
            "dataset_name": "events",
            "output_file": EVENTS_PARQUET,
            "description": "All Retailrocket visitor-item events converted from CSV to Parquet.",
        },
        {
            "dataset_name": "category_tree",
            "output_file": CATEGORY_TREE_PARQUET,
            "description": "Retailrocket category hierarchy converted from CSV to Parquet.",
        },
        {
            "dataset_name": "item_properties_selected",
            "output_file": ITEM_PROPERTIES_SELECTED_PARQUET,
            "description": "Selected item properties limited to categoryid and available.",
        },
        {
            "dataset_name": "item_category_latest",
            "output_file": ITEM_CATEGORY_LATEST_PARQUET,
            "description": "Latest categoryid value per item.",
        },
        {
            "dataset_name": "item_availability_latest",
            "output_file": ITEM_AVAILABILITY_LATEST_PARQUET,
            "description": "Latest availability value per item.",
        },
    ]

    rows: list[dict[str, Any]] = []

    for output in outputs:
        output_file = output["output_file"]

        rows.append(
            {
                "dataset_name": output["dataset_name"],
                "output_file": str(output_file),
                "description": output["description"],
                "row_count": count_parquet_rows(connection, output_file),
                "column_count": count_parquet_columns(connection, output_file),
                "file_size_mb": round(output_file.stat().st_size / (1024 * 1024), 3),
                "status": "PASS" if output_file.exists() else "FAIL",
            }
        )

    return pd.DataFrame(rows)


def prepare_events(connection: duckdb.DuckDBPyConnection) -> None:
    sql = f"""
    COPY (
        SELECT
            CAST(timestamp AS BIGINT) AS event_timestamp_ms,
            to_timestamp(CAST(timestamp AS DOUBLE) / 1000) AS event_datetime,
            CAST(visitorid AS BIGINT) AS visitor_id,
            CAST(event AS VARCHAR) AS event_type,
            CAST(itemid AS BIGINT) AS item_id,
            TRY_CAST(transactionid AS BIGINT) AS transaction_id,
            'retailrocket' AS _source_system,
            '{EVENTS_CSV}' AS _source_file
        FROM read_csv_auto('{EVENTS_CSV}', header = true)
    )
    TO '{EVENTS_PARQUET}'
    (FORMAT PARQUET);
    """

    run_sql(connection, sql)


def prepare_category_tree(connection: duckdb.DuckDBPyConnection) -> None:
    sql = f"""
    COPY (
        SELECT
            CAST(categoryid AS BIGINT) AS category_id,
            TRY_CAST(parentid AS BIGINT) AS parent_category_id,
            CASE
                WHEN parentid IS NULL THEN 1
                ELSE 0
            END AS is_root_category,
            'retailrocket' AS _source_system,
            '{CATEGORY_TREE_CSV}' AS _source_file
        FROM read_csv_auto('{CATEGORY_TREE_CSV}', header = true)
    )
    TO '{CATEGORY_TREE_PARQUET}'
    (FORMAT PARQUET);
    """

    run_sql(connection, sql)


def prepare_selected_item_properties(connection: duckdb.DuckDBPyConnection) -> None:
    """
    Extracts only categoryid and available from the very large item property files.

    This avoids pivoting all 1,104 properties and keeps the first Retailrocket
    staged layer practical for local development.
    """
    sql = f"""
    COPY (
        WITH item_properties AS (
            SELECT
                CAST(timestamp AS BIGINT) AS property_timestamp_ms,
                to_timestamp(CAST(timestamp AS DOUBLE) / 1000) AS property_datetime,
                CAST(itemid AS BIGINT) AS item_id,
                CAST(property AS VARCHAR) AS property_name,
                CAST(value AS VARCHAR) AS property_value,
                'item_properties_part1.csv' AS _source_part
            FROM read_csv(
                '{ITEM_PROPERTIES_PART1_CSV}',
                header = true,
                columns = {{
                    'timestamp': 'BIGINT',
                    'itemid': 'BIGINT',
                    'property': 'VARCHAR',
                    'value': 'VARCHAR'
                }}
            )
            WHERE property IN ('categoryid', 'available')

            UNION ALL

            SELECT
                CAST(timestamp AS BIGINT) AS property_timestamp_ms,
                to_timestamp(CAST(timestamp AS DOUBLE) / 1000) AS property_datetime,
                CAST(itemid AS BIGINT) AS item_id,
                CAST(property AS VARCHAR) AS property_name,
                CAST(value AS VARCHAR) AS property_value,
                'item_properties_part2.csv' AS _source_part
            FROM read_csv(
                '{ITEM_PROPERTIES_PART2_CSV}',
                header = true,
                columns = {{
                    'timestamp': 'BIGINT',
                    'itemid': 'BIGINT',
                    'property': 'VARCHAR',
                    'value': 'VARCHAR'
                }}
            )
            WHERE property IN ('categoryid', 'available')
        )
        SELECT
            property_timestamp_ms,
            property_datetime,
            item_id,
            property_name,
            property_value,
            _source_part,
            'retailrocket' AS _source_system
        FROM item_properties
    )
    TO '{ITEM_PROPERTIES_SELECTED_PARQUET}'
    (FORMAT PARQUET);
    """

    run_sql(connection, sql)


def prepare_latest_item_category(connection: duckdb.DuckDBPyConnection) -> None:
    sql = f"""
    COPY (
        WITH ranked_categories AS (
            SELECT
                item_id,
                TRY_CAST(property_value AS BIGINT) AS category_id,
                property_timestamp_ms,
                property_datetime,
                ROW_NUMBER() OVER (
                    PARTITION BY item_id
                    ORDER BY property_timestamp_ms DESC
                ) AS row_number
            FROM read_parquet('{ITEM_PROPERTIES_SELECTED_PARQUET}')
            WHERE property_name = 'categoryid'
        )
        SELECT
            item_id,
            category_id,
            property_timestamp_ms AS category_timestamp_ms,
            property_datetime AS category_datetime,
            'retailrocket' AS _source_system,
            '{ITEM_PROPERTIES_SELECTED_PARQUET}' AS _source_file
        FROM ranked_categories
        WHERE row_number = 1
    )
    TO '{ITEM_CATEGORY_LATEST_PARQUET}'
    (FORMAT PARQUET);
    """

    run_sql(connection, sql)


def prepare_latest_item_availability(connection: duckdb.DuckDBPyConnection) -> None:
    sql = f"""
    COPY (
        WITH ranked_availability AS (
            SELECT
                item_id,
                property_value AS available_raw_value,
                CASE
                    WHEN LOWER(property_value) IN ('1', 'true') THEN 1
                    WHEN LOWER(property_value) IN ('0', 'false') THEN 0
                    ELSE NULL
                END AS is_available,
                property_timestamp_ms,
                property_datetime,
                ROW_NUMBER() OVER (
                    PARTITION BY item_id
                    ORDER BY property_timestamp_ms DESC
                ) AS row_number
            FROM read_parquet('{ITEM_PROPERTIES_SELECTED_PARQUET}')
            WHERE property_name = 'available'
        )
        SELECT
            item_id,
            available_raw_value,
            is_available,
            property_timestamp_ms AS availability_timestamp_ms,
            property_datetime AS availability_datetime,
            'retailrocket' AS _source_system,
            '{ITEM_PROPERTIES_SELECTED_PARQUET}' AS _source_file
        FROM ranked_availability
        WHERE row_number = 1
    )
    TO '{ITEM_AVAILABILITY_LATEST_PARQUET}'
    (FORMAT PARQUET);
    """

    run_sql(connection, sql)


def main() -> None:
    ensure_input_files_exist()

    STAGED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with duckdb.connect(database=":memory:") as connection:
        print("Preparing Retailrocket events...")
        prepare_events(connection)

        print("Preparing Retailrocket category tree...")
        prepare_category_tree(connection)

        print("Preparing selected Retailrocket item properties...")
        prepare_selected_item_properties(connection)

        print("Preparing latest item category table...")
        prepare_latest_item_category(connection)

        print("Preparing latest item availability table...")
        prepare_latest_item_availability(connection)

        summary_df = build_summary(connection)

    summary_df.to_csv(REPORT_PATH, index=False)

    print("\nRetailrocket preparation completed.")
    print(f"Preparation summary saved: {REPORT_PATH}")
    print()
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()