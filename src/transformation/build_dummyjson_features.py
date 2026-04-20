from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


WAREHOUSE_PATH = Path("data/warehouse/recommart.duckdb")
SQL_PATH = Path("sql/dummyjson_features.sql")
FEATURE_DIR = Path("data/features/source=dummyjson")
REPORT_PATH = Path("reports/dummyjson_feature_summary.csv")


FEATURE_TABLES = [
    ("features", "dummyjson_user_features", FEATURE_DIR / "user_features.parquet"),
    ("features", "dummyjson_item_features", FEATURE_DIR / "item_features.parquet"),
    ("features", "dummyjson_interaction_features", FEATURE_DIR / "interaction_features.parquet"),
]


def read_sql_file(sql_path: Path) -> str:
    if not sql_path.exists():
        raise FileNotFoundError(f"SQL file not found: {sql_path}")

    return sql_path.read_text(encoding="utf-8")


def ensure_prerequisites() -> None:
    if not WAREHOUSE_PATH.exists():
        raise FileNotFoundError(
            f"DuckDB warehouse not found: {WAREHOUSE_PATH}. "
            "Run python -m src.transformation.load_dummyjson_to_duckdb first."
        )

    if not SQL_PATH.exists():
        raise FileNotFoundError(f"Feature SQL file not found: {SQL_PATH}")

    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


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


def get_null_metadata_count(connection: duckdb.DuckDBPyConnection) -> int:
    """
    Counts interactions where catalog metadata was missing.
    This is expected because our product catalog may not include every cart product.
    """
    return int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM features.dummyjson_interaction_features
            WHERE has_catalog_metadata = 0
            """
        ).fetchone()[0]
    )


def build_summary(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for schema_name, table_name, output_path in FEATURE_TABLES:
        relation_type = get_relation_type(connection, schema_name, table_name)

        if relation_type == "MISSING":
            rows.append(
                {
                    "schema_name": schema_name,
                    "table_name": table_name,
                    "relation_type": relation_type,
                    "output_file": str(output_path),
                    "row_count": None,
                    "column_count": None,
                    "output_exists": output_path.exists(),
                    "status": "FAIL",
                }
            )
            continue

        row_count = get_row_count(connection, schema_name, table_name)
        column_count = get_column_count(connection, schema_name, table_name)

        rows.append(
            {
                "schema_name": schema_name,
                "table_name": table_name,
                "relation_type": relation_type,
                "output_file": str(output_path),
                "row_count": row_count,
                "column_count": column_count,
                "output_exists": output_path.exists(),
                "status": "PASS" if output_path.exists() and row_count > 0 else "FAIL",
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    ensure_prerequisites()

    sql_text = read_sql_file(SQL_PATH)

    with duckdb.connect(str(WAREHOUSE_PATH)) as connection:
        connection.execute(sql_text)

        summary_df = build_summary(connection)
        missing_metadata_count = get_null_metadata_count(connection)

        summary_df["missing_catalog_metadata_interactions"] = None
        summary_df.loc[
            summary_df["table_name"] == "dummyjson_interaction_features",
            "missing_catalog_metadata_interactions",
        ] = missing_metadata_count

        summary_df.to_csv(REPORT_PATH, index=False)

    print("DummyJSON feature build completed.")
    print(f"Warehouse path: {WAREHOUSE_PATH}")
    print(f"Feature SQL executed: {SQL_PATH}")
    print(f"Feature directory: {FEATURE_DIR}")
    print(f"Feature summary saved: {REPORT_PATH}")
    print()
    print(summary_df.to_string(index=False))

    failed_count = int((summary_df["status"] == "FAIL").sum())

    if failed_count > 0:
        raise RuntimeError(f"Feature build completed with {failed_count} failed checks.")


if __name__ == "__main__":
    main()