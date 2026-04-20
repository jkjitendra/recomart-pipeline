-- DuckDB warehouse setup for DummyJSON staged data.
-- This SQL file creates staged tables and recommendation-oriented mart views.

CREATE SCHEMA IF NOT EXISTS staged;
CREATE SCHEMA IF NOT EXISTS mart;

-- ---------------------------------------------------------------------
-- Staged tables
-- ---------------------------------------------------------------------

CREATE OR REPLACE TABLE staged.dummyjson_products AS
SELECT *
FROM read_parquet('data/staged/source=dummyjson/products.parquet', hive_partitioning = false);

CREATE OR REPLACE TABLE staged.dummyjson_users AS
SELECT *
FROM read_parquet('data/staged/source=dummyjson/users.parquet', hive_partitioning = false);

CREATE OR REPLACE TABLE staged.dummyjson_carts AS
SELECT *
FROM read_parquet('data/staged/source=dummyjson/carts.parquet', hive_partitioning = false);

CREATE OR REPLACE TABLE staged.dummyjson_cart_items AS
SELECT *
FROM read_parquet('data/staged/source=dummyjson/cart_items.parquet', hive_partitioning = false);

-- ---------------------------------------------------------------------
-- Mart views for recommendation pipeline
-- ---------------------------------------------------------------------

CREATE OR REPLACE VIEW mart.dummyjson_interactions AS
SELECT
    user_id,
    product_id AS item_id,
    cart_id,
    'cart_purchase' AS event_type,
    CAST(quantity AS DOUBLE) AS interaction_weight,
    price,
    total AS line_total,
    discounted_total AS line_discounted_total,
    _source_system,
    _source_file,
    _prepared_at
FROM staged.dummyjson_cart_items;

CREATE OR REPLACE VIEW mart.dummyjson_item_features AS
SELECT
    id AS item_id,
    title,
    description,
    category,
    price,
    discountPercentage AS discount_percentage,
    rating,
    stock,
    brand,
    sku,
    availabilityStatus AS availability_status,
    minimumOrderQuantity AS minimum_order_quantity,
    _source_system,
    _source_file,
    _prepared_at
FROM staged.dummyjson_products;

CREATE OR REPLACE VIEW mart.dummyjson_user_cart_features AS
SELECT
    user_id,
    COUNT(DISTINCT cart_id) AS cart_count,
    SUM(total) AS total_cart_value,
    SUM(discounted_total) AS total_discounted_cart_value,
    SUM(total_products) AS total_products_in_carts,
    SUM(total_quantity) AS total_quantity_in_carts,
    AVG(total) AS avg_cart_value,
    AVG(total_quantity) AS avg_cart_quantity
FROM staged.dummyjson_carts
GROUP BY user_id;