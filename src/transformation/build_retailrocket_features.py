from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


WAREHOUSE_PATH = Path("data/warehouse/recomart.duckdb")
SQL_PATH = Path("sql/retailrocket_features.sql")

FEATURE_DIR = Path("data/features/source=retailrocket")
USER_FEATURES_PATH = FEATURE_DIR / "user_features.parquet"
ITEM_FEATURES_PATH = FEATURE_DIR / "item_features.parquet"
USER_ITEM_FEATURES_PATH = FEATURE_DIR / "user_item_features.parquet"

REPORT_PATH = Path("reports/retailrocket_feature_summary.csv")
SQL_SCHEMA_SUMMARY_PATH = Path("reports/sql_schema_summary.csv")


FEATURE_TABLES = [
    {
        "schema_name": "features",
        "table_name": "retailrocket_user_features",
        "output_file": USER_FEATURES_PATH,
    },
    {
        "schema_name": "features",
        "table_name": "retailrocket_item_features",
        "output_file": ITEM_FEATURES_PATH,
    },
    {
        "schema_name": "features",
        "table_name": "retailrocket_user_item_features",
        "output_file": USER_ITEM_FEATURES_PATH,
    },
]


def ensure_prerequisites() -> None:
    if not WAREHOUSE_PATH.exists():
        raise FileNotFoundError(
            f"Warehouse not found: {WAREHOUSE_PATH}. "
            "Run python -m src.transformation.load_retailrocket_to_duckdb first."
        )

    if not SQL_PATH.exists():
        raise FileNotFoundError(f"Feature SQL file not found: {SQL_PATH}")

    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_row_count(
    connection: duckdb.DuckDBPyConnection,
    schema_name: str,
    table_name: str,
) -> int:
    return int(
        connection.execute(
            f"SELECT COUNT(*) FROM {schema_name}.{table_name}"
        ).fetchone()[0]
    )


def get_column_count(
    connection: duckdb.DuckDBPyConnection,
    schema_name: str,
    table_name: str,
) -> int:
    result = connection.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_schema = ?
          AND table_name = ?
        """,
        [schema_name, table_name],
    ).fetchone()

    return int(result[0])


def export_table_to_parquet(
    connection: duckdb.DuckDBPyConnection,
    schema_name: str,
    table_name: str,
    output_file: Path,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    connection.execute(
        f"""
        COPY (
            SELECT *
            FROM {schema_name}.{table_name}
        )
        TO '{output_file}'
        (FORMAT PARQUET);
        """
    )


def build_summary(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for feature_table in FEATURE_TABLES:
        schema_name = feature_table["schema_name"]
        table_name = feature_table["table_name"]
        output_file = feature_table["output_file"]

        export_table_to_parquet(
            connection=connection,
            schema_name=schema_name,
            table_name=table_name,
            output_file=output_file,
        )

        row_count = get_row_count(connection, schema_name, table_name)
        column_count = get_column_count(connection, schema_name, table_name)

        rows.append(
            {
                "schema_name": schema_name,
                "table_name": table_name,
                "output_file": str(output_file),
                "row_count": row_count,
                "column_count": column_count,
                "file_size_mb": round(output_file.stat().st_size / (1024 * 1024), 3),
                "status": "PASS" if output_file.exists() and row_count > 0 else "FAIL",
            }
        )

    return pd.DataFrame(rows)


def build_sql_schema_summary(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    query = """
        SELECT
            table_schema AS schema_name,
            table_name,
            column_name,
            ordinal_position,
            data_type,
            is_nullable
        FROM information_schema.columns
        WHERE table_schema IN ('staged', 'curated', 'mart', 'features')
        ORDER BY table_schema, table_name, ordinal_position
    """
    return connection.execute(query).fetchdf()


def main() -> None:
    ensure_prerequisites()

    sql_text = SQL_PATH.read_text(encoding="utf-8")

    with duckdb.connect(str(WAREHOUSE_PATH)) as connection:
        connection.execute(sql_text)
        summary_df = build_summary(connection)
        schema_summary_df = build_sql_schema_summary(connection)

    summary_df.to_csv(REPORT_PATH, index=False)
    schema_summary_df.to_csv(SQL_SCHEMA_SUMMARY_PATH, index=False)

    print("Retailrocket feature build completed.")
    print(f"Warehouse path: {WAREHOUSE_PATH}")
    print(f"Feature SQL executed: {SQL_PATH}")
    print(f"Feature directory: {FEATURE_DIR}")
    print(f"Feature summary saved: {REPORT_PATH}")
    print(f"SQL schema summary saved: {SQL_SCHEMA_SUMMARY_PATH}")
    print()
    print(summary_df.to_string(index=False))

    failed_count = int((summary_df["status"] == "FAIL").sum())

    if failed_count > 0:
        raise RuntimeError(f"Retailrocket feature build had {failed_count} failed outputs.")


if __name__ == "__main__":
    main()
