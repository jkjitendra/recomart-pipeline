-- Feature engineering SQL for DummyJSON recommendation pipeline.
-- This file creates model-ready feature tables from DuckDB mart views.

CREATE SCHEMA IF NOT EXISTS features;

-- ---------------------------------------------------------------------
-- User-level features
-- ---------------------------------------------------------------------

CREATE OR REPLACE TABLE features.dummyjson_user_features AS
WITH interaction_agg AS (
    SELECT
        user_id,
        COUNT(*) AS user_total_interactions,
        COUNT(DISTINCT item_id) AS user_unique_items,
        SUM(interaction_weight) AS user_total_quantity,
        SUM(line_total) AS user_total_revenue,
        AVG(interaction_weight) AS user_avg_quantity_per_interaction,
        AVG(line_total) AS user_avg_line_total
    FROM mart.dummyjson_interactions
    GROUP BY user_id
)
SELECT
    u.user_id,
    u.cart_count,
    u.total_cart_value,
    u.total_discounted_cart_value,
    u.total_products_in_carts,
    u.total_quantity_in_carts,
    u.avg_cart_value,
    u.avg_cart_quantity,
    COALESCE(i.user_total_interactions, 0) AS user_total_interactions,
    COALESCE(i.user_unique_items, 0) AS user_unique_items,
    COALESCE(i.user_total_quantity, 0) AS user_total_quantity,
    COALESCE(i.user_total_revenue, 0) AS user_total_revenue,
    COALESCE(i.user_avg_quantity_per_interaction, 0) AS user_avg_quantity_per_interaction,
    COALESCE(i.user_avg_line_total, 0) AS user_avg_line_total
FROM mart.dummyjson_user_cart_features u
LEFT JOIN interaction_agg i
    ON u.user_id = i.user_id;

-- ---------------------------------------------------------------------
-- Item-level features
-- ---------------------------------------------------------------------

CREATE OR REPLACE TABLE features.dummyjson_item_features AS
WITH item_interaction_agg AS (
    SELECT
        item_id,
        COUNT(*) AS item_total_interactions,
        COUNT(DISTINCT user_id) AS item_distinct_user_count,
        SUM(interaction_weight) AS item_total_quantity,
        SUM(line_total) AS item_total_revenue,
        AVG(interaction_weight) AS item_avg_quantity_per_interaction,
        AVG(line_total) AS item_avg_line_total
    FROM mart.dummyjson_interactions
    GROUP BY item_id
)
SELECT
    item.item_id,
    item.title,
    item.description,
    item.category,
    item.price,
    item.discount_percentage,
    item.rating,
    item.stock,
    item.brand,
    item.sku,
    item.availability_status,
    item.minimum_order_quantity,
    COALESCE(agg.item_total_interactions, 0) AS item_total_interactions,
    COALESCE(agg.item_distinct_user_count, 0) AS item_distinct_user_count,
    COALESCE(agg.item_total_quantity, 0) AS item_total_quantity,
    COALESCE(agg.item_total_revenue, 0) AS item_total_revenue,
    COALESCE(agg.item_avg_quantity_per_interaction, 0) AS item_avg_quantity_per_interaction,
    COALESCE(agg.item_avg_line_total, 0) AS item_avg_line_total,
    item._source_system,
    item._source_file,
    item._prepared_at
FROM mart.dummyjson_item_features item
LEFT JOIN item_interaction_agg agg
    ON item.item_id = agg.item_id;

-- ---------------------------------------------------------------------
-- Interaction-level features
-- ---------------------------------------------------------------------

CREATE OR REPLACE TABLE features.dummyjson_interaction_features AS
SELECT
    interaction.user_id,
    interaction.item_id,
    interaction.cart_id,
    interaction.event_type,
    interaction.interaction_weight,
    interaction.price AS interaction_price,
    interaction.line_total,
    interaction.line_discounted_total,
    COALESCE(item.category, 'unknown') AS item_category,
    item.rating AS item_rating,
    item.stock AS item_stock,
    item.brand AS item_brand,
    CASE
        WHEN item.item_id IS NULL THEN 0
        ELSE 1
    END AS has_catalog_metadata,
    interaction._source_system,
    interaction._source_file,
    interaction._prepared_at
FROM mart.dummyjson_interactions interaction
LEFT JOIN mart.dummyjson_item_features item
    ON interaction.item_id = item.item_id;

-- ---------------------------------------------------------------------
-- Export feature tables to Parquet
-- ---------------------------------------------------------------------

COPY features.dummyjson_user_features
TO 'data/features/source=dummyjson/user_features.parquet'
(FORMAT PARQUET);

COPY features.dummyjson_item_features
TO 'data/features/source=dummyjson/item_features.parquet'
(FORMAT PARQUET);

COPY features.dummyjson_interaction_features
TO 'data/features/source=dummyjson/interaction_features.parquet'
(FORMAT PARQUET);