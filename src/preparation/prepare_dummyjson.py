import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


RAW_DIR = Path("data/raw/source=dummyjson")
STAGED_DIR = Path("data/staged/source=dummyjson")
REPORT_PATH = Path("reports/dummyjson_preparation_summary.csv")


def find_latest_file(dataset_type: str) -> Path:
    """
    Finds the latest raw JSON file for a given dataset type.
    Looks under:
    data/raw/source=dummyjson/type=products/

    Returns the most recently modified JSON file.
    """
    search_dir = RAW_DIR / f"type={dataset_type}"
    json_files = sorted(search_dir.rglob("*.json"), key=lambda path: path.stat().st_mtime)

    if not json_files:
        raise FileNotFoundError(f"No raw JSON files found for dataset type: {dataset_type}")

    return json_files[-1]


def load_json(file_path: Path) -> dict[str, Any]:
    """
    Reads a JSON file and returns it as a Python dictionary.
    """
    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def add_metadata_columns(df: pd.DataFrame, source_file: Path) -> pd.DataFrame:
    """
    Adds lineage metadata to each staged table.

    These columns help us understand:
    - where the data came from
    - when it was prepared
    - which source system produced it
    """
    df = df.copy()
    df["_source_system"] = "dummyjson"
    df["_source_file"] = str(source_file)
    df["_prepared_at"] = datetime.now().isoformat(timespec="seconds")
    return df


def prepare_products() -> tuple[pd.DataFrame, Path]:
    """
    Converts raw products JSON into a flat products DataFrame.
    """
    source_file = find_latest_file("products")
    raw_data = load_json(source_file)

    products = raw_data.get("products", [])
    products_df = pd.json_normalize(products)

    products_df = add_metadata_columns(products_df, source_file)

    return products_df, source_file


def prepare_users() -> tuple[pd.DataFrame, Path]:
    """
    Converts raw users JSON into a flat users DataFrame.
    """
    source_file = find_latest_file("users")
    raw_data = load_json(source_file)

    users = raw_data.get("users", [])
    users_df = pd.json_normalize(users)

    users_df = add_metadata_columns(users_df, source_file)

    return users_df, source_file


def prepare_carts() -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """
    Converts raw carts JSON into two tables:

    1. carts_df:
       One row per cart.

    2. cart_items_df:
       One row per product inside each cart.
    """
    source_file = find_latest_file("carts")
    raw_data = load_json(source_file)

    carts = raw_data.get("carts", [])

    cart_rows = []
    cart_item_rows = []

    for cart in carts:
        cart_id = cart.get("id")
        user_id = cart.get("userId")

        cart_rows.append(
            {
                "cart_id": cart_id,
                "user_id": user_id,
                "total": cart.get("total"),
                "discounted_total": cart.get("discountedTotal"),
                "total_products": cart.get("totalProducts"),
                "total_quantity": cart.get("totalQuantity"),
            }
        )

        for product in cart.get("products", []):
            cart_item_rows.append(
                {
                    "cart_id": cart_id,
                    "user_id": user_id,
                    "product_id": product.get("id"),
                    "product_title": product.get("title"),
                    "price": product.get("price"),
                    "quantity": product.get("quantity"),
                    "total": product.get("total"),
                    "discount_percentage": product.get("discountPercentage"),
                    "discounted_total": product.get("discountedTotal"),
                    "thumbnail": product.get("thumbnail"),
                }
            )

    carts_df = pd.DataFrame(cart_rows)
    cart_items_df = pd.DataFrame(cart_item_rows)

    carts_df = add_metadata_columns(carts_df, source_file)
    cart_items_df = add_metadata_columns(cart_items_df, source_file)

    return carts_df, cart_items_df, source_file


def save_parquet(df: pd.DataFrame, output_path: Path) -> None:
    """
    Saves a DataFrame as a Parquet file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)


def add_summary_row(
    summary_rows: list[dict[str, Any]],
    dataset_name: str,
    source_file: Path,
    output_file: Path,
    df: pd.DataFrame,
) -> None:
    """
    Adds one row to the preparation summary report.
    """
    summary_rows.append(
        {
            "dataset_name": dataset_name,
            "source_file": str(source_file),
            "output_file": str(output_file),
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": ", ".join(df.columns),
        }
    )


def main() -> None:
    summary_rows: list[dict[str, Any]] = []

    products_df, products_source = prepare_products()
    users_df, users_source = prepare_users()
    carts_df, cart_items_df, carts_source = prepare_carts()

    products_output = STAGED_DIR / "products.parquet"
    users_output = STAGED_DIR / "users.parquet"
    carts_output = STAGED_DIR / "carts.parquet"
    cart_items_output = STAGED_DIR / "cart_items.parquet"

    save_parquet(products_df, products_output)
    save_parquet(users_df, users_output)
    save_parquet(carts_df, carts_output)
    save_parquet(cart_items_df, cart_items_output)

    add_summary_row(summary_rows, "products", products_source, products_output, products_df)
    add_summary_row(summary_rows, "users", users_source, users_output, users_df)
    add_summary_row(summary_rows, "carts", carts_source, carts_output, carts_df)
    add_summary_row(summary_rows, "cart_items", carts_source, cart_items_output, cart_items_df)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(REPORT_PATH, index=False)

    print("DummyJSON preparation completed.")
    print(f"Products shape: {products_df.shape}")
    print(f"Users shape: {users_df.shape}")
    print(f"Carts shape: {carts_df.shape}")
    print(f"Cart items shape: {cart_items_df.shape}")
    print(f"Preparation summary saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()