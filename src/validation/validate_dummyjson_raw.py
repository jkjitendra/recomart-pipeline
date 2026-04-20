import json
from pathlib import Path
from typing import Any

import pandas as pd


RAW_DIR = Path("data/raw/source=dummyjson")
REPORT_PATH = Path("reports/data_quality/dummyjson_raw_validation_report.csv")


EXPECTED_TOP_LEVEL_KEYS = {
    "products": {"products", "total", "skip", "limit"},
    "users": {"users", "total", "skip", "limit"},
    "carts": {"carts", "total", "skip", "limit"},
}


REQUIRED_RECORD_FIELDS = {
    "products": {"id", "title", "price", "category"},
    "users": {"id", "firstName", "lastName", "email"},
    "carts": {"id", "products", "total", "userId"},
}


def get_dataset_type(file_path: Path) -> str:
    """
    Extracts dataset type from folder name.

    Example path:
    data/raw/source=dummyjson/type=products/ingest_date=2026-04-20/products_....json

    This function returns:
    products
    """
    for part in file_path.parts:
        if part.startswith("type="):
            return part.replace("type=", "")

    return "unknown"


def load_json(file_path: Path) -> dict[str, Any]:
    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def add_result(
    results: list[dict[str, Any]],
    file_path: Path,
    dataset_type: str,
    check_name: str,
    status: str,
    details: str,
) -> None:
    results.append(
        {
            "file_path": str(file_path),
            "dataset_type": dataset_type,
            "check_name": check_name,
            "status": status,
            "details": details,
        }
    )


def validate_top_level_keys(
    data: dict[str, Any],
    file_path: Path,
    dataset_type: str,
    results: list[dict[str, Any]],
) -> None:
    actual_keys = set(data.keys())
    expected_keys = EXPECTED_TOP_LEVEL_KEYS.get(dataset_type)

    if expected_keys is None:
        add_result(
            results,
            file_path,
            dataset_type,
            "top_level_schema",
            "FAIL",
            f"Unknown dataset type: {dataset_type}",
        )
        return

    missing_keys = expected_keys - actual_keys

    if missing_keys:
        add_result(
            results,
            file_path,
            dataset_type,
            "top_level_schema",
            "FAIL",
            f"Missing top-level keys: {sorted(missing_keys)}",
        )
    else:
        add_result(
            results,
            file_path,
            dataset_type,
            "top_level_schema",
            "PASS",
            f"Found expected keys: {sorted(expected_keys)}",
        )


def validate_record_list(
    data: dict[str, Any],
    file_path: Path,
    dataset_type: str,
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records = data.get(dataset_type)

    if not isinstance(records, list):
        add_result(
            results,
            file_path,
            dataset_type,
            "record_list_exists",
            "FAIL",
            f"Expected key '{dataset_type}' to contain a list",
        )
        return []

    if len(records) == 0:
        add_result(
            results,
            file_path,
            dataset_type,
            "record_count",
            "FAIL",
            "Record list is empty",
        )
    else:
        add_result(
            results,
            file_path,
            dataset_type,
            "record_count",
            "PASS",
            f"Found {len(records)} records",
        )

    return records


def validate_required_fields(
    records: list[dict[str, Any]],
    file_path: Path,
    dataset_type: str,
    results: list[dict[str, Any]],
) -> None:
    required_fields = REQUIRED_RECORD_FIELDS.get(dataset_type, set())

    if not records:
        add_result(
            results,
            file_path,
            dataset_type,
            "required_fields",
            "FAIL",
            "No records available for required field validation",
        )
        return

    missing_summary = {}

    for field in required_fields:
        missing_count = sum(
            1
            for record in records
            if field not in record or record.get(field) is None
        )

        if missing_count > 0:
            missing_summary[field] = missing_count

    if missing_summary:
        add_result(
            results,
            file_path,
            dataset_type,
            "required_fields",
            "FAIL",
            f"Missing/null required fields: {missing_summary}",
        )
    else:
        add_result(
            results,
            file_path,
            dataset_type,
            "required_fields",
            "PASS",
            f"All records contain required fields: {sorted(required_fields)}",
        )


def validate_duplicate_ids(
    records: list[dict[str, Any]],
    file_path: Path,
    dataset_type: str,
    results: list[dict[str, Any]],
) -> None:
    ids = [record.get("id") for record in records if "id" in record]
    duplicate_count = len(ids) - len(set(ids))

    if duplicate_count > 0:
        add_result(
            results,
            file_path,
            dataset_type,
            "duplicate_ids",
            "FAIL",
            f"Found {duplicate_count} duplicate ids",
        )
    else:
        add_result(
            results,
            file_path,
            dataset_type,
            "duplicate_ids",
            "PASS",
            "No duplicate ids found",
        )


def validate_products(
    records: list[dict[str, Any]],
    file_path: Path,
    results: list[dict[str, Any]],
) -> None:
    invalid_price_count = sum(
        1
        for record in records
        if record.get("price") is None or record.get("price", 0) < 0
    )

    if invalid_price_count > 0:
        add_result(
            results,
            file_path,
            "products",
            "product_price_range",
            "FAIL",
            f"Found {invalid_price_count} products with missing/negative price",
        )
    else:
        add_result(
            results,
            file_path,
            "products",
            "product_price_range",
            "PASS",
            "All product prices are present and non-negative",
        )


def validate_users(
    records: list[dict[str, Any]],
    file_path: Path,
    results: list[dict[str, Any]],
) -> None:
    invalid_email_count = sum(
        1
        for record in records
        if not isinstance(record.get("email"), str) or "@" not in record.get("email", "")
    )

    if invalid_email_count > 0:
        add_result(
            results,
            file_path,
            "users",
            "user_email_format",
            "FAIL",
            f"Found {invalid_email_count} users with invalid email format",
        )
    else:
        add_result(
            results,
            file_path,
            "users",
            "user_email_format",
            "PASS",
            "All user emails contain @",
        )


def validate_carts(
    records: list[dict[str, Any]],
    file_path: Path,
    results: list[dict[str, Any]],
) -> None:
    empty_cart_count = sum(
        1
        for record in records
        if not isinstance(record.get("products"), list) or len(record.get("products", [])) == 0
    )

    if empty_cart_count > 0:
        add_result(
            results,
            file_path,
            "carts",
            "cart_products_not_empty",
            "FAIL",
            f"Found {empty_cart_count} carts with no products",
        )
    else:
        add_result(
            results,
            file_path,
            "carts",
            "cart_products_not_empty",
            "PASS",
            "All carts contain at least one product",
        )


def validate_file(file_path: Path, results: list[dict[str, Any]]) -> None:
    dataset_type = get_dataset_type(file_path)
    data = load_json(file_path)

    validate_top_level_keys(data, file_path, dataset_type, results)
    records = validate_record_list(data, file_path, dataset_type, results)

    validate_required_fields(records, file_path, dataset_type, results)
    validate_duplicate_ids(records, file_path, dataset_type, results)

    if dataset_type == "products":
        validate_products(records, file_path, results)
    elif dataset_type == "users":
        validate_users(records, file_path, results)
    elif dataset_type == "carts":
        validate_carts(records, file_path, results)


def main() -> None:
    json_files = sorted(RAW_DIR.rglob("*.json"))

    if not json_files:
        print(f"No JSON files found under {RAW_DIR}")
        return

    results: list[dict[str, Any]] = []

    for file_path in json_files:
        validate_file(file_path, results)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    report_df = pd.DataFrame(results)
    report_df.to_csv(REPORT_PATH, index=False)

    total_checks = len(report_df)
    failed_checks = int((report_df["status"] == "FAIL").sum())
    passed_checks = int((report_df["status"] == "PASS").sum())

    print(f"Validation report saved: {REPORT_PATH}")
    print(f"Total checks: {total_checks}")
    print(f"Passed checks: {passed_checks}")
    print(f"Failed checks: {failed_checks}")

    if failed_checks > 0:
        print("\nFailed checks:")
        print(report_df[report_df["status"] == "FAIL"].to_string(index=False))
    else:
        print("\nAll DummyJSON raw validation checks passed.")


if __name__ == "__main__":
    main()