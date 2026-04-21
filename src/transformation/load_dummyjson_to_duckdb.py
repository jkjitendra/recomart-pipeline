from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


WAREHOUSE_PATH = Path("data/warehouse/recomart.duckdb")
SQL_PATH = Path("sql/dummyjson_warehouse.sql")
REPORT_PATH = Path("reports/duckdb_load_summary.csv")


RELATIONS_TO_CHECK = [
    ("staged", "dummyjson_products"),
    ("staged", "dummyjson_users"),
    ("staged", "dummyjson_carts"),
    ("staged", "dummyjson_cart_items"),
    ("mart", "dummyjson_interactions"),
    ("mart", "dummyjson_item_features"),
    ("mart", "dummyjson_user_cart_features"),
]


def read_sql_file(sql_path: Path) -> str:
    if not sql_path.exists():
        raise FileNotFoundError(f"SQL file not found: {sql_path}")

    return sql_path.read_text(encoding="utf-8")


def get_relation_type(
    connection: duckdb.DuckDBPyConnection,
    schema_name: str,
    relation_name: str,
) -> str:
    result = connection.execute(
        """
        SELECT table_type
        FROM information_schema.tables
        WHERE table_schema = ?
          AND table_name = ?
        """,
        [schema_name, relation_name],
    ).fetchone()

    if result is None:
        return "MISSING"

    return str(result[0])


def get_row_count(
    connection: duckdb.DuckDBPyConnection,
    schema_name: str,
    relation_name: str,
) -> int:
    query = f"SELECT COUNT(*) FROM {schema_name}.{relation_name}"
    return int(connection.execute(query).fetchone()[0])


def get_column_count(
    connection: duckdb.DuckDBPyConnection,
    schema_name: str,
    relation_name: str,
) -> int:
    result = connection.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_schema = ?
          AND table_name = ?
        """,
        [schema_name, relation_name],
    ).fetchone()

    return int(result[0])


def build_summary(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for schema_name, relation_name in RELATIONS_TO_CHECK:
        relation_type = get_relation_type(connection, schema_name, relation_name)

        if relation_type == "MISSING":
            rows.append(
                {
                    "schema_name": schema_name,
                    "relation_name": relation_name,
                    "relation_type": relation_type,
                    "row_count": None,
                    "column_count": None,
                    "status": "FAIL",
                }
            )
            continue

        row_count = get_row_count(connection, schema_name, relation_name)
        column_count = get_column_count(connection, schema_name, relation_name)

        rows.append(
            {
                "schema_name": schema_name,
                "relation_name": relation_name,
                "relation_type": relation_type,
                "row_count": row_count,
                "column_count": column_count,
                "status": "PASS",
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    WAREHOUSE_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    sql_text = read_sql_file(SQL_PATH)

    with duckdb.connect(str(WAREHOUSE_PATH)) as connection:
        connection.execute(sql_text)

        summary_df = build_summary(connection)
        summary_df.to_csv(REPORT_PATH, index=False)

    print("DuckDB warehouse load completed.")
    print(f"Warehouse path: {WAREHOUSE_PATH}")
    print(f"SQL file executed: {SQL_PATH}")
    print(f"Load summary saved: {REPORT_PATH}")
    print()
    print(summary_df.to_string(index=False))

    failed_count = int((summary_df["status"] == "FAIL").sum())

    if failed_count > 0:
        raise RuntimeError(f"DuckDB load completed with {failed_count} failed relation checks.")


if __name__ == "__main__":
    main()