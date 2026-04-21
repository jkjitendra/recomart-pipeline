from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


WAREHOUSE_PATH = Path("data/warehouse/recommart.duckdb")
SQL_PATH = Path("sql/retailrocket_warehouse.sql")
REPORT_PATH = Path("reports/retailrocket_duckdb_load_summary.csv")


RELATIONS = [
    ("staged", "retailrocket_events"),
    ("staged", "retailrocket_category_tree"),
    ("staged", "retailrocket_item_properties_selected"),
    ("staged", "retailrocket_item_category_latest"),
    ("staged", "retailrocket_item_availability_latest"),
    ("mart", "retailrocket_interactions"),
    ("mart", "retailrocket_user_item_interactions"),
    ("mart", "retailrocket_user_features"),
    ("mart", "retailrocket_item_features"),
]


def ensure_prerequisites() -> None:
    if not SQL_PATH.exists():
        raise FileNotFoundError(f"SQL file not found: {SQL_PATH}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    WAREHOUSE_PATH.parent.mkdir(parents=True, exist_ok=True)


def read_sql() -> str:
    return SQL_PATH.read_text(encoding="utf-8")


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
    return int(
        connection.execute(
            f"SELECT COUNT(*) FROM {schema_name}.{relation_name}"
        ).fetchone()[0]
    )


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

    for schema_name, relation_name in RELATIONS:
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
                "status": "PASS" if row_count > 0 else "FAIL",
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    ensure_prerequisites()

    sql_text = read_sql()

    with duckdb.connect(str(WAREHOUSE_PATH)) as connection:
        connection.execute(sql_text)
        summary_df = build_summary(connection)

    summary_df.to_csv(REPORT_PATH, index=False)

    print("Retailrocket DuckDB warehouse load completed.")
    print(f"Warehouse path: {WAREHOUSE_PATH}")
    print(f"SQL file executed: {SQL_PATH}")
    print(f"Load summary saved: {REPORT_PATH}")
    print()
    print(summary_df.to_string(index=False))

    failed_count = int((summary_df["status"] == "FAIL").sum())

    if failed_count > 0:
        raise RuntimeError(f"Retailrocket warehouse load had {failed_count} failed relations.")


if __name__ == "__main__":
    main()