from pathlib import Path
from typing import Any

import pandas as pd


STAGED_DIR = Path("data/staged/source=dummyjson")
REPORT_PATH = Path("reports/data_quality/dummyjson_staged_validation_report.csv")


EXPECTED_FILES = {
    "products": STAGED_DIR / "products.parquet",
    "users": STAGED_DIR / "users.parquet",
    "carts": STAGED_DIR / "carts.parquet",
    "cart_items": STAGED_DIR / "cart_items.parquet",
}


REQUIRED_COLUMNS = {
    "products": {"id", "title", "category", "price", "_source_system", "_source_file", "_prepared_at"},
    "users": {"id", "firstName", "lastName", "email", "_source_system", "_source_file", "_prepared_at"},
    "carts": {"cart_id", "user_id", "total", "discounted_total", "total_products", "total_quantity"},
    "cart_items": {"cart_id", "user_id", "product_id", "product_title", "price", "quantity", "total"},
}


def add_result(
    results: list[dict[str, Any]],
    dataset_name: str,
    check_name: str,
    status: str,
    details: str,
) -> None:
    results.append(
        {
            "dataset_name": dataset_name,
            "check_name": check_name,
            "status": status,
            "details": details,
        }
    )


def check_file_exists(
    dataset_name: str,
    file_path: Path,
    results: list[dict[str, Any]],
) -> bool:
    if file_path.exists():
        add_result(
            results,
            dataset_name,
            "file_exists",
            "PASS",
            f"Found file: {file_path}",
        )
        return True

    add_result(
        results,
        dataset_name,
        "file_exists",
        "FAIL",
        f"Missing file: {file_path}",
    )
    return False


def load_parquet(
    dataset_name: str,
    file_path: Path,
    results: list[dict[str, Any]],
) -> pd.DataFrame | None:
    try:
        df = pd.read_parquet(file_path)
        add_result(
            results,
            dataset_name,
            "read_parquet",
            "PASS",
            f"Successfully read file with shape {df.shape}",
        )
        return df

    except Exception as exc:
        add_result(
            results,
            dataset_name,
            "read_parquet",
            "FAIL",
            f"Failed to read Parquet file: {exc}",
        )
        return None


def validate_row_count(
    dataset_name: str,
    df: pd.DataFrame,
    results: list[dict[str, Any]],
) -> None:
    if len(df) > 0:
        add_result(
            results,
            dataset_name,
            "row_count",
            "PASS",
            f"Found {len(df)} rows",
        )
    else:
        add_result(
            results,
            dataset_name,
            "row_count",
            "FAIL",
            "Dataset has zero rows",
        )


def validate_required_columns(
    dataset_name: str,
    df: pd.DataFrame,
    results: list[dict[str, Any]],
) -> None:
    expected_columns = REQUIRED_COLUMNS.get(dataset_name, set())
    actual_columns = set(df.columns)
    missing_columns = expected_columns - actual_columns

    if missing_columns:
        add_result(
            results,
            dataset_name,
            "required_columns",
            "FAIL",
            f"Missing required columns: {sorted(missing_columns)}",
        )
    else:
        add_result(
            results,
            dataset_name,
            "required_columns",
            "PASS",
            f"All required columns found: {sorted(expected_columns)}",
        )


def validate_unique_column(
    dataset_name: str,
    df: pd.DataFrame,
    column_name: str,
    results: list[dict[str, Any]],
) -> None:
    if column_name not in df.columns:
        add_result(
            results,
            dataset_name,
            f"unique_{column_name}",
            "FAIL",
            f"Column not found: {column_name}",
        )
        return

    duplicate_count = int(df[column_name].duplicated().sum())

    if duplicate_count == 0:
        add_result(
            results,
            dataset_name,
            f"unique_{column_name}",
            "PASS",
            f"No duplicate values found in {column_name}",
        )
    else:
        add_result(
            results,
            dataset_name,
            f"unique_{column_name}",
            "FAIL",
            f"Found {duplicate_count} duplicate values in {column_name}",
        )


def validate_no_nulls(
    dataset_name: str,
    df: pd.DataFrame,
    columns: list[str],
    results: list[dict[str, Any]],
) -> None:
    for column in columns:
        if column not in df.columns:
            add_result(
                results,
                dataset_name,
                f"not_null_{column}",
                "FAIL",
                f"Column not found: {column}",
            )
            continue

        null_count = int(df[column].isna().sum())

        if null_count == 0:
            add_result(
                results,
                dataset_name,
                f"not_null_{column}",
                "PASS",
                f"No nulls found in {column}",
            )
        else:
            add_result(
                results,
                dataset_name,
                f"not_null_{column}",
                "FAIL",
                f"Found {null_count} null values in {column}",
            )


def validate_non_negative(
    dataset_name: str,
    df: pd.DataFrame,
    column_name: str,
    results: list[dict[str, Any]],
) -> None:
    if column_name not in df.columns:
        add_result(
            results,
            dataset_name,
            f"non_negative_{column_name}",
            "FAIL",
            f"Column not found: {column_name}",
        )
        return

    invalid_count = int((df[column_name] < 0).sum())

    if invalid_count == 0:
        add_result(
            results,
            dataset_name,
            f"non_negative_{column_name}",
            "PASS",
            f"All values in {column_name} are non-negative",
        )
    else:
        add_result(
            results,
            dataset_name,
            f"non_negative_{column_name}",
            "FAIL",
            f"Found {invalid_count} negative values in {column_name}",
        )


def validate_positive(
    dataset_name: str,
    df: pd.DataFrame,
    column_name: str,
    results: list[dict[str, Any]],
) -> None:
    if column_name not in df.columns:
        add_result(
            results,
            dataset_name,
            f"positive_{column_name}",
            "FAIL",
            f"Column not found: {column_name}",
        )
        return

    invalid_count = int((df[column_name] <= 0).sum())

    if invalid_count == 0:
        add_result(
            results,
            dataset_name,
            f"positive_{column_name}",
            "PASS",
            f"All values in {column_name} are positive",
        )
    else:
        add_result(
            results,
            dataset_name,
            f"positive_{column_name}",
            "FAIL",
            f"Found {invalid_count} non-positive values in {column_name}",
        )


def validate_between(
    dataset_name: str,
    df: pd.DataFrame,
    column_name: str,
    minimum: float,
    maximum: float,
    results: list[dict[str, Any]],
) -> None:
    if column_name not in df.columns:
        add_result(
            results,
            dataset_name,
            f"range_{column_name}",
            "FAIL",
            f"Column not found: {column_name}",
        )
        return

    invalid_count = int(((df[column_name] < minimum) | (df[column_name] > maximum)).sum())

    if invalid_count == 0:
        add_result(
            results,
            dataset_name,
            f"range_{column_name}",
            "PASS",
            f"All values in {column_name} are between {minimum} and {maximum}",
        )
    else:
        add_result(
            results,
            dataset_name,
            f"range_{column_name}",
            "FAIL",
            f"Found {invalid_count} values outside range {minimum}-{maximum}",
        )


def validate_email_format(
    users_df: pd.DataFrame,
    results: list[dict[str, Any]],
) -> None:
    dataset_name = "users"

    if "email" not in users_df.columns:
        add_result(
            results,
            dataset_name,
            "email_format",
            "FAIL",
            "Column not found: email",
        )
        return

    invalid_count = int(
        users_df["email"]
        .fillna("")
        .astype(str)
        .apply(lambda value: "@" not in value)
        .sum()
    )

    if invalid_count == 0:
        add_result(
            results,
            dataset_name,
            "email_format",
            "PASS",
            "All emails contain @",
        )
    else:
        add_result(
            results,
            dataset_name,
            "email_format",
            "FAIL",
            f"Found {invalid_count} invalid emails",
        )


def validate_referential_integrity(
    tables: dict[str, pd.DataFrame],
    results: list[dict[str, Any]],
) -> None:
    carts_df = tables["carts"]
    cart_items_df = tables["cart_items"]
    users_df = tables["users"]
    products_df = tables["products"]

    missing_cart_ids = set(cart_items_df["cart_id"]) - set(carts_df["cart_id"])

    if missing_cart_ids:
        add_result(
            results,
            "cart_items",
            "cart_id_exists_in_carts",
            "FAIL",
            f"Found {len(missing_cart_ids)} cart_ids in cart_items that do not exist in carts",
        )
    else:
        add_result(
            results,
            "cart_items",
            "cart_id_exists_in_carts",
            "PASS",
            "All cart_items.cart_id values exist in carts.cart_id",
        )

    missing_user_ids = set(cart_items_df["user_id"]) - set(users_df["id"])

    if missing_user_ids:
        add_result(
            results,
            "cart_items",
            "user_id_exists_in_users",
            "FAIL",
            f"Found {len(missing_user_ids)} user_ids in cart_items that do not exist in users",
        )
    else:
        add_result(
            results,
            "cart_items",
            "user_id_exists_in_users",
            "PASS",
            "All cart_items.user_id values exist in users.id",
        )

    missing_product_ids = set(cart_items_df["product_id"]) - set(products_df["id"])
    total_distinct_cart_products = len(set(cart_items_df["product_id"]))
    matched_product_count = total_distinct_cart_products - len(missing_product_ids)

    if missing_product_ids:
        add_result(
            results,
            "cart_items",
            "product_id_exists_in_products",
            "WARN",
            (
                f"{matched_product_count}/{total_distinct_cart_products} distinct cart product_ids "
                f"exist in products.parquet. Missing {len(missing_product_ids)} product_ids. "
                "This can happen because only the first 100 products were fetched from DummyJSON."
            ),
        )
    else:
        add_result(
            results,
            "cart_items",
            "product_id_exists_in_products",
            "PASS",
            "All cart_items.product_id values exist in products.id",
        )


def validate_products(products_df: pd.DataFrame, results: list[dict[str, Any]]) -> None:
    dataset_name = "products"

    validate_unique_column(dataset_name, products_df, "id", results)
    validate_no_nulls(dataset_name, products_df, ["id", "title", "category", "price"], results)
    validate_non_negative(dataset_name, products_df, "price", results)

    if "rating" in products_df.columns:
        validate_between(dataset_name, products_df, "rating", 0, 5, results)


def validate_users(users_df: pd.DataFrame, results: list[dict[str, Any]]) -> None:
    dataset_name = "users"

    validate_unique_column(dataset_name, users_df, "id", results)
    validate_no_nulls(dataset_name, users_df, ["id", "firstName", "lastName", "email"], results)
    validate_email_format(users_df, results)

    if "age" in users_df.columns:
        validate_between(dataset_name, users_df, "age", 0, 120, results)


def validate_carts(carts_df: pd.DataFrame, results: list[dict[str, Any]]) -> None:
    dataset_name = "carts"

    validate_unique_column(dataset_name, carts_df, "cart_id", results)
    validate_no_nulls(dataset_name, carts_df, ["cart_id", "user_id", "total"], results)
    validate_non_negative(dataset_name, carts_df, "total", results)
    validate_positive(dataset_name, carts_df, "total_products", results)
    validate_positive(dataset_name, carts_df, "total_quantity", results)


def validate_cart_items(cart_items_df: pd.DataFrame, results: list[dict[str, Any]]) -> None:
    dataset_name = "cart_items"

    validate_no_nulls(dataset_name, cart_items_df, ["cart_id", "user_id", "product_id", "quantity"], results)
    validate_positive(dataset_name, cart_items_df, "quantity", results)
    validate_non_negative(dataset_name, cart_items_df, "price", results)
    validate_non_negative(dataset_name, cart_items_df, "total", results)


def main() -> None:
    results: list[dict[str, Any]] = []
    tables: dict[str, pd.DataFrame] = {}

    for dataset_name, file_path in EXPECTED_FILES.items():
        exists = check_file_exists(dataset_name, file_path, results)

        if not exists:
            continue

        df = load_parquet(dataset_name, file_path, results)

        if df is None:
            continue

        tables[dataset_name] = df

        validate_row_count(dataset_name, df, results)
        validate_required_columns(dataset_name, df, results)

    if "products" in tables:
        validate_products(tables["products"], results)

    if "users" in tables:
        validate_users(tables["users"], results)

    if "carts" in tables:
        validate_carts(tables["carts"], results)

    if "cart_items" in tables:
        validate_cart_items(tables["cart_items"], results)

    required_for_referential_checks = {"products", "users", "carts", "cart_items"}

    if required_for_referential_checks.issubset(tables.keys()):
        validate_referential_integrity(tables, results)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    report_df = pd.DataFrame(results)
    report_df.to_csv(REPORT_PATH, index=False)

    total_checks = len(report_df)
    passed_checks = int((report_df["status"] == "PASS").sum())
    failed_checks = int((report_df["status"] == "FAIL").sum())
    warning_checks = int((report_df["status"] == "WARN").sum())

    print(f"Staged validation report saved: {REPORT_PATH}")
    print(f"Total checks: {total_checks}")
    print(f"Passed checks: {passed_checks}")
    print(f"Warnings: {warning_checks}")
    print(f"Failed checks: {failed_checks}")

    if failed_checks > 0:
        print("\nFailed checks:")
        print(report_df[report_df["status"] == "FAIL"].to_string(index=False))

    if warning_checks > 0:
        print("\nWarning checks:")
        print(report_df[report_df["status"] == "WARN"].to_string(index=False))

    if failed_checks == 0:
        print("\nNo failed staged validation checks.")


if __name__ == "__main__":
    main()