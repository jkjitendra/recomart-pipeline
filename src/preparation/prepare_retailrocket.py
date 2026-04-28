from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_BATCH_DIR = PROJECT_ROOT / "data/raw/source=retailrocket_batch"
RAW_API_DIR = PROJECT_ROOT / "data/raw/source=retailrocket_api"
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

PREPARATION_REPORT_PATH = PROJECT_ROOT / "reports/retailrocket_preparation_summary.csv"
CURATED_REPORT_PATH = PROJECT_ROOT / "reports/retailrocket_curated_summary.csv"


def display_path(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)


def latest_ingestion_dir(raw_source_dir: Path, data_type: str) -> Path:
    type_dir = raw_source_dir / f"type={data_type}"

    if not type_dir.exists():
        raise FileNotFoundError(
            f"Missing raw type directory: {display_path(type_dir)}. "
            "Run the Retailrocket ingestion scripts first."
        )

    candidates = sorted(
        path for path in type_dir.iterdir()
        if path.is_dir() and path.name.startswith("ingestion_timestamp=")
    )

    if not candidates:
        raise FileNotFoundError(f"No ingestion_timestamp partitions found in {display_path(type_dir)}")

    return candidates[-1]


def all_ingestion_dirs(raw_source_dir: Path, data_type: str) -> list[Path]:
    type_dir = raw_source_dir / f"type={data_type}"

    if not type_dir.exists():
        raise FileNotFoundError(
            f"Missing raw type directory: {display_path(type_dir)}. "
            "Run the Retailrocket ingestion scripts first."
        )

    candidates = sorted(
        path for path in type_dir.iterdir()
        if path.is_dir() and path.name.startswith("ingestion_timestamp=")
    )

    if not candidates:
        raise FileNotFoundError(f"No ingestion_timestamp partitions found in {display_path(type_dir)}")

    return candidates


def first_existing(path_candidates: list[Path]) -> Path:
    for path in path_candidates:
        if path.exists():
            return path

    raise FileNotFoundError(f"None of the expected files exist: {[display_path(path) for path in path_candidates]}")


def read_metadata(ingestion_dir: Path) -> dict[str, Any]:
    metadata_path = ingestion_dir / "_metadata.json"
    if not metadata_path.exists():
        return {}

    return json.loads(metadata_path.read_text(encoding="utf-8"))


def resolve_raw_inputs() -> dict[str, dict[str, Any]]:
    events_dir = latest_ingestion_dir(RAW_BATCH_DIR, "events")
    item_part1_dir = latest_ingestion_dir(RAW_BATCH_DIR, "item_properties_part1")
    category_tree_dir = latest_ingestion_dir(RAW_BATCH_DIR, "category_tree")
    api_delta_dirs = all_ingestion_dirs(RAW_API_DIR, "item_properties_delta")
    api_delta_files = [
        first_existing(
            [
                api_delta_dir / "item_properties_delta.parquet",
                api_delta_dir / "item_properties_delta.csv",
                api_delta_dir / "item_properties_delta.json",
            ]
        )
        for api_delta_dir in api_delta_dirs
    ]

    inputs = {
        "events": {
            "data_type": "events",
            "path": first_existing([events_dir / "events.csv"]),
            "ingestion_dir": events_dir,
        },
        "item_properties_part1": {
            "data_type": "item_properties_part1",
            "path": first_existing([item_part1_dir / "item_properties_part1.csv"]),
            "ingestion_dir": item_part1_dir,
        },
        "category_tree": {
            "data_type": "category_tree",
            "path": first_existing([category_tree_dir / "category_tree.csv"]),
            "ingestion_dir": category_tree_dir,
        },
        "item_properties_api_delta": {
            "data_type": "item_properties_delta",
            "paths": api_delta_files,
            "ingestion_dirs": api_delta_dirs,
            "partition_count": len(api_delta_dirs),
        },
    }

    for item in inputs.values():
        if "ingestion_dir" in item:
            item["metadata"] = read_metadata(item["ingestion_dir"])
        else:
            item["metadata"] = [read_metadata(path) for path in item["ingestion_dirs"]]

    return inputs


def run_sql(connection: duckdb.DuckDBPyConnection, sql: str) -> None:
    connection.execute(sql)


def parquet_relation(path: Path) -> str:
    return f"read_parquet('{path}', hive_partitioning = false)"


def count_parquet_rows(connection: duckdb.DuckDBPyConnection, parquet_path: Path) -> int:
    return int(connection.execute(f"SELECT COUNT(*) FROM {parquet_relation(parquet_path)}").fetchone()[0])


def count_parquet_columns(connection: duckdb.DuckDBPyConnection, parquet_path: Path) -> int:
    return len(connection.execute(f"SELECT * FROM {parquet_relation(parquet_path)} LIMIT 0").fetchdf().columns)


def prepare_events(connection: duckdb.DuckDBPyConnection, events_csv: Path) -> None:
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
            '{display_path(events_csv)}' AS _source_file
        FROM read_csv_auto('{events_csv}', header = true)
    )
    TO '{EVENTS_PARQUET}'
    (FORMAT PARQUET);
    """

    run_sql(connection, sql)


def prepare_category_tree(connection: duckdb.DuckDBPyConnection, category_tree_csv: Path) -> None:
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
            '{display_path(category_tree_csv)}' AS _source_file
        FROM read_csv_auto('{category_tree_csv}', header = true)
    )
    TO '{CATEGORY_TREE_PARQUET}'
    (FORMAT PARQUET);
    """

    run_sql(connection, sql)


def prepare_item_properties_batch(connection: duckdb.DuckDBPyConnection, item_properties_csv: Path) -> None:
    sql = f"""
    COPY (
        SELECT
            CAST(timestamp AS BIGINT) AS property_timestamp_ms,
            to_timestamp(CAST(timestamp AS DOUBLE) / 1000) AS property_datetime,
            CAST(itemid AS BIGINT) AS item_id,
            CAST(property AS VARCHAR) AS property_name,
            CAST(value AS VARCHAR) AS property_value,
            'item_properties_part1' AS source_feed,
            'retailrocket_batch' AS source_system,
            '{display_path(item_properties_csv)}' AS source_file
        FROM read_csv(
            '{item_properties_csv}',
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
    TO '{ITEM_PROPERTIES_BATCH_PARQUET}'
    (FORMAT PARQUET);
    """

    run_sql(connection, sql)


def source_relation_for_file(api_delta_file: Path) -> str:
    if api_delta_file.suffix == ".parquet":
        return parquet_relation(api_delta_file)
    if api_delta_file.suffix == ".csv":
        return f"read_csv_auto('{api_delta_file}', header = true)"
    if api_delta_file.suffix == ".json":
        return f"read_json_auto('{api_delta_file}')"
    raise ValueError(f"Unsupported API delta file type: {api_delta_file}")


def prepare_item_properties_api_delta(connection: duckdb.DuckDBPyConnection, api_delta_files: list[Path]) -> None:
    select_statements = []

    for api_delta_file in api_delta_files:
        source_relation = source_relation_for_file(api_delta_file)
        select_statements.append(
            f"""
            SELECT
                CAST(timestamp AS BIGINT) AS property_timestamp_ms,
                to_timestamp(CAST(timestamp AS DOUBLE) / 1000) AS property_datetime,
                CAST(itemid AS BIGINT) AS item_id,
                CAST(property AS VARCHAR) AS property_name,
                CAST(value AS VARCHAR) AS property_value,
                'item_properties_delta' AS source_feed,
                'retailrocket_api' AS source_system,
                '{display_path(api_delta_file)}' AS source_file,
                CAST(ingestion_timestamp AS VARCHAR) AS ingestion_timestamp
            FROM {source_relation}
            """
        )

    union_sql = "\nUNION ALL\n".join(select_statements)

    sql = f"""
    COPY (
        {union_sql}
    )
    TO '{ITEM_PROPERTIES_API_DELTA_PARQUET}'
    (FORMAT PARQUET);
    """

    run_sql(connection, sql)


def prepare_item_properties_combined(connection: duckdb.DuckDBPyConnection) -> None:
    sql = f"""
    COPY (
        SELECT
            property_timestamp_ms,
            property_datetime,
            item_id,
            property_name,
            property_value,
            source_feed,
            source_system,
            source_file,
            NULL::VARCHAR AS ingestion_timestamp
        FROM {parquet_relation(ITEM_PROPERTIES_BATCH_PARQUET)}

        UNION ALL

        SELECT
            property_timestamp_ms,
            property_datetime,
            item_id,
            property_name,
            property_value,
            source_feed,
            source_system,
            source_file,
            ingestion_timestamp
        FROM {parquet_relation(ITEM_PROPERTIES_API_DELTA_PARQUET)}
    )
    TO '{ITEM_PROPERTIES_COMBINED_PARQUET}'
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
                source_system,
                source_file,
                ROW_NUMBER() OVER (
                    PARTITION BY item_id
                    ORDER BY property_timestamp_ms DESC, source_system DESC
                ) AS row_number
            FROM {parquet_relation(ITEM_PROPERTIES_COMBINED_PARQUET)}
            WHERE property_name = 'categoryid'
        )
        SELECT
            item_id,
            category_id,
            property_timestamp_ms AS category_timestamp_ms,
            property_datetime AS category_datetime,
            source_system AS _source_system,
            source_file AS _source_file
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
                source_system,
                source_file,
                ROW_NUMBER() OVER (
                    PARTITION BY item_id
                    ORDER BY property_timestamp_ms DESC, source_system DESC
                ) AS row_number
            FROM {parquet_relation(ITEM_PROPERTIES_COMBINED_PARQUET)}
            WHERE property_name = 'available'
        )
        SELECT
            item_id,
            available_raw_value,
            is_available,
            property_timestamp_ms AS availability_timestamp_ms,
            property_datetime AS availability_datetime,
            source_system AS _source_system,
            source_file AS _source_file
        FROM ranked_availability
        WHERE row_number = 1
    )
    TO '{ITEM_AVAILABILITY_LATEST_PARQUET}'
    (FORMAT PARQUET);
    """

    run_sql(connection, sql)


def prepare_curated_interactions(connection: duckdb.DuckDBPyConnection) -> None:
    sql = f"""
    COPY (
        SELECT
            e.visitor_id AS user_id,
            e.item_id,
            e.event_type,
            e.event_timestamp_ms,
            e.event_datetime,
            e.transaction_id,
            CASE
                WHEN e.event_type = 'view' THEN 1.0
                WHEN e.event_type = 'addtocart' THEN 3.0
                WHEN e.event_type = 'transaction' THEN 5.0
                ELSE 0.0
            END AS interaction_weight,
            c.category_id,
            t.parent_category_id,
            a.is_available,
            a.available_raw_value,
            CASE WHEN c.item_id IS NOT NULL THEN 1 ELSE 0 END AS has_category_metadata,
            CASE WHEN a.item_id IS NOT NULL THEN 1 ELSE 0 END AS has_availability_metadata
        FROM {parquet_relation(EVENTS_PARQUET)} e
        LEFT JOIN {parquet_relation(ITEM_CATEGORY_LATEST_PARQUET)} c
            ON e.item_id = c.item_id
        LEFT JOIN {parquet_relation(CATEGORY_TREE_PARQUET)} t
            ON c.category_id = t.category_id
        LEFT JOIN {parquet_relation(ITEM_AVAILABILITY_LATEST_PARQUET)} a
            ON e.item_id = a.item_id
    )
    TO '{CURATED_INTERACTIONS_PARQUET}'
    (FORMAT PARQUET);
    """

    run_sql(connection, sql)


def prepare_curated_items(connection: duckdb.DuckDBPyConnection) -> None:
    sql = f"""
    COPY (
        WITH item_event_agg AS (
            SELECT
                item_id,
                COUNT(*) AS total_events,
                COUNT(DISTINCT visitor_id) AS unique_users,
                SUM(CASE WHEN event_type = 'view' THEN 1 ELSE 0 END) AS view_events,
                SUM(CASE WHEN event_type = 'addtocart' THEN 1 ELSE 0 END) AS addtocart_events,
                SUM(CASE WHEN event_type = 'transaction' THEN 1 ELSE 0 END) AS transaction_events,
                MIN(event_timestamp_ms) AS first_event_timestamp_ms,
                MAX(event_timestamp_ms) AS last_event_timestamp_ms,
                MIN(event_datetime) AS first_event_datetime,
                MAX(event_datetime) AS last_event_datetime
            FROM {parquet_relation(EVENTS_PARQUET)}
            GROUP BY item_id
        )
        SELECT
            item.item_id,
            item.total_events,
            item.unique_users,
            item.view_events,
            item.addtocart_events,
            item.transaction_events,
            item.first_event_timestamp_ms,
            item.last_event_timestamp_ms,
            item.first_event_datetime,
            item.last_event_datetime,
            c.category_id,
            t.parent_category_id,
            a.is_available,
            a.available_raw_value,
            CASE WHEN c.item_id IS NOT NULL THEN 1 ELSE 0 END AS has_category_metadata,
            CASE WHEN a.item_id IS NOT NULL THEN 1 ELSE 0 END AS has_availability_metadata
        FROM item_event_agg item
        LEFT JOIN {parquet_relation(ITEM_CATEGORY_LATEST_PARQUET)} c
            ON item.item_id = c.item_id
        LEFT JOIN {parquet_relation(CATEGORY_TREE_PARQUET)} t
            ON c.category_id = t.category_id
        LEFT JOIN {parquet_relation(ITEM_AVAILABILITY_LATEST_PARQUET)} a
            ON item.item_id = a.item_id
    )
    TO '{CURATED_ITEMS_PARQUET}'
    (FORMAT PARQUET);
    """

    run_sql(connection, sql)


def prepare_curated_user_item_interactions(connection: duckdb.DuckDBPyConnection) -> None:
    sql = f"""
    COPY (
        SELECT
            user_id,
            item_id,
            COUNT(*) AS total_events,
            SUM(CASE WHEN event_type = 'view' THEN 1 ELSE 0 END) AS view_count,
            SUM(CASE WHEN event_type = 'addtocart' THEN 1 ELSE 0 END) AS addtocart_count,
            SUM(CASE WHEN event_type = 'transaction' THEN 1 ELSE 0 END) AS transaction_count,
            SUM(interaction_weight) AS interaction_score,
            MAX(interaction_weight) AS max_event_weight,
            MIN(event_timestamp_ms) AS first_event_timestamp_ms,
            MAX(event_timestamp_ms) AS last_event_timestamp_ms,
            MIN(event_datetime) AS first_event_datetime,
            MAX(event_datetime) AS last_event_datetime,
            MAX(category_id) AS category_id,
            MAX(parent_category_id) AS parent_category_id,
            MAX(is_available) AS is_available,
            MAX(has_category_metadata) AS has_category_metadata,
            MAX(has_availability_metadata) AS has_availability_metadata,
            CASE WHEN SUM(CASE WHEN event_type = 'transaction' THEN 1 ELSE 0 END) > 0 THEN 1 ELSE 0 END AS has_transaction,
            CASE WHEN SUM(CASE WHEN event_type = 'addtocart' THEN 1 ELSE 0 END) > 0 THEN 1 ELSE 0 END AS has_addtocart
        FROM {parquet_relation(CURATED_INTERACTIONS_PARQUET)}
        GROUP BY user_id, item_id
    )
    TO '{CURATED_USER_ITEM_INTERACTIONS_PARQUET}'
    (FORMAT PARQUET);
    """

    run_sql(connection, sql)


def build_summary(
    connection: duckdb.DuckDBPyConnection,
    outputs: list[dict[str, Any]],
    api_delta_metrics: dict[str, Any] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for output in outputs:
        output_file = output["output_file"]
        row = {
            "dataset_name": output["dataset_name"],
            "output_file": display_path(output_file),
            "description": output["description"],
            "source_files": output.get("source_files", ""),
            "row_count": count_parquet_rows(connection, output_file),
            "column_count": count_parquet_columns(connection, output_file),
            "file_size_mb": round(output_file.stat().st_size / (1024 * 1024), 3),
            "status": "PASS" if output_file.exists() else "FAIL",
        }

        if api_delta_metrics and output["dataset_name"] == "item_properties_api_delta":
            row.update(api_delta_metrics)
        elif api_delta_metrics:
            row.update(
                {
                    "api_delta_partition_count": None,
                    "api_delta_raw_rows": None,
                    "api_delta_rows_after_deduplication": None,
                    "api_delta_deduplication_key": None,
                }
            )

        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    raw_inputs = resolve_raw_inputs()

    STAGED_DIR.mkdir(parents=True, exist_ok=True)
    CURATED_DIR.mkdir(parents=True, exist_ok=True)
    PREPARATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with duckdb.connect(database=":memory:") as connection:
        print("Preparing Retailrocket events from raw batch landing...")
        prepare_events(connection, raw_inputs["events"]["path"])

        print("Preparing Retailrocket category tree from raw batch landing...")
        prepare_category_tree(connection, raw_inputs["category_tree"]["path"])

        print("Preparing Retailrocket item_properties_part1 from raw batch landing...")
        prepare_item_properties_batch(connection, raw_inputs["item_properties_part1"]["path"])

        print("Preparing Retailrocket item_properties_delta from raw API landing...")
        prepare_item_properties_api_delta(connection, raw_inputs["item_properties_api_delta"]["paths"])

        print("Preparing combined Retailrocket item properties...")
        prepare_item_properties_combined(connection)

        print("Preparing latest item category table...")
        prepare_latest_item_category(connection)

        print("Preparing latest item availability table...")
        prepare_latest_item_availability(connection)

        print("Preparing curated interactions...")
        prepare_curated_interactions(connection)

        print("Preparing curated items...")
        prepare_curated_items(connection)

        print("Preparing curated user-item interactions...")
        prepare_curated_user_item_interactions(connection)

        staged_outputs = [
            {
                "dataset_name": "events",
                "output_file": EVENTS_PARQUET,
                "description": "Retailrocket visitor-item events prepared from raw batch ingestion.",
                "source_files": display_path(raw_inputs["events"]["path"]),
            },
            {
                "dataset_name": "category_tree",
                "output_file": CATEGORY_TREE_PARQUET,
                "description": "Retailrocket category hierarchy prepared from raw batch ingestion.",
                "source_files": display_path(raw_inputs["category_tree"]["path"]),
            },
            {
                "dataset_name": "item_properties_batch",
                "output_file": ITEM_PROPERTIES_BATCH_PARQUET,
                "description": "Selected Retailrocket batch item properties from raw item_properties_part1.",
                "source_files": display_path(raw_inputs["item_properties_part1"]["path"]),
            },
            {
                "dataset_name": "item_properties_api_delta",
                "output_file": ITEM_PROPERTIES_API_DELTA_PARQUET,
                "description": "Retailrocket item property delta records from all REST/mock API raw landings.",
                "source_files": "; ".join(
                    display_path(path)
                    for path in raw_inputs["item_properties_api_delta"]["paths"]
                ),
            },
            {
                "dataset_name": "item_properties_combined",
                "output_file": ITEM_PROPERTIES_COMBINED_PARQUET,
                "description": "Combined batch and API delta item properties.",
            },
            {
                "dataset_name": "item_category_latest",
                "output_file": ITEM_CATEGORY_LATEST_PARQUET,
                "description": "Latest categoryid value per item from combined item properties.",
            },
            {
                "dataset_name": "item_availability_latest",
                "output_file": ITEM_AVAILABILITY_LATEST_PARQUET,
                "description": "Latest availability value per item from combined item properties.",
            },
        ]

        curated_outputs = [
            {
                "dataset_name": "curated_interactions",
                "output_file": CURATED_INTERACTIONS_PARQUET,
                "description": "EDA-ready Retailrocket interaction rows enriched with latest item metadata.",
            },
            {
                "dataset_name": "curated_items",
                "output_file": CURATED_ITEMS_PARQUET,
                "description": "EDA-ready Retailrocket item dataset with event aggregates and metadata.",
            },
            {
                "dataset_name": "curated_user_item_interactions",
                "output_file": CURATED_USER_ITEM_INTERACTIONS_PARQUET,
                "description": "EDA/modeling-ready user-item interaction aggregates.",
            },
        ]

        api_delta_raw_rows = count_parquet_rows(connection, ITEM_PROPERTIES_API_DELTA_PARQUET)
        api_delta_metrics = {
            "api_delta_partition_count": raw_inputs["item_properties_api_delta"]["partition_count"],
            "api_delta_raw_rows": api_delta_raw_rows,
            "api_delta_rows_after_deduplication": api_delta_raw_rows,
            "api_delta_deduplication_key": "none - all API delta rows are preserved",
        }

        staged_summary_df = build_summary(connection, staged_outputs, api_delta_metrics=api_delta_metrics)
        curated_summary_df = build_summary(connection, curated_outputs)

    staged_summary_df.to_csv(PREPARATION_REPORT_PATH, index=False)
    curated_summary_df.to_csv(CURATED_REPORT_PATH, index=False)

    print("\nRetailrocket preparation completed.")
    print(f"Preparation summary saved: {display_path(PREPARATION_REPORT_PATH)}")
    print(f"Curated summary saved: {display_path(CURATED_REPORT_PATH)}")
    print()
    print(staged_summary_df.to_string(index=False))
    print()
    print(curated_summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
