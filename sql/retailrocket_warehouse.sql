-- Retailrocket DuckDB warehouse transformation.
-- Loads staged Retailrocket Parquet files and creates recommender mart tables.

CREATE SCHEMA IF NOT EXISTS staged;
CREATE SCHEMA IF NOT EXISTS mart;

-- ---------------------------------------------------------------------
-- Staged tables
-- ---------------------------------------------------------------------

CREATE OR REPLACE TABLE staged.retailrocket_events AS
SELECT *
FROM read_parquet('data/staged/source=retailrocket/events.parquet', hive_partitioning = false);

CREATE OR REPLACE TABLE staged.retailrocket_category_tree AS
SELECT *
FROM read_parquet('data/staged/source=retailrocket/category_tree.parquet', hive_partitioning = false);

CREATE OR REPLACE TABLE staged.retailrocket_item_properties_selected AS
SELECT *
FROM read_parquet('data/staged/source=retailrocket/item_properties_selected.parquet', hive_partitioning = false);

CREATE OR REPLACE TABLE staged.retailrocket_item_category_latest AS
SELECT *
FROM read_parquet('data/staged/source=retailrocket/item_category_latest.parquet', hive_partitioning = false);

CREATE OR REPLACE TABLE staged.retailrocket_item_availability_latest AS
SELECT *
FROM read_parquet('data/staged/source=retailrocket/item_availability_latest.parquet', hive_partitioning = false);

-- ---------------------------------------------------------------------
-- Interaction-level mart
-- ---------------------------------------------------------------------

CREATE OR REPLACE VIEW mart.retailrocket_interactions AS
SELECT
    visitor_id AS user_id,
    item_id,
    event_type,
    event_timestamp_ms,
    event_datetime,
    transaction_id,
    CASE
        WHEN event_type = 'view' THEN 1.0
        WHEN event_type = 'addtocart' THEN 3.0
        WHEN event_type = 'transaction' THEN 5.0
        ELSE 0.0
    END AS interaction_weight,
    _source_system,
    _source_file
FROM staged.retailrocket_events;

-- ---------------------------------------------------------------------
-- User-item aggregate mart
-- One row per user-item pair.
-- This becomes the core training table for recommendation models.
-- ---------------------------------------------------------------------

CREATE OR REPLACE TABLE mart.retailrocket_user_item_interactions AS
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
    CASE
        WHEN SUM(CASE WHEN event_type = 'transaction' THEN 1 ELSE 0 END) > 0 THEN 1
        ELSE 0
    END AS has_transaction,
    CASE
        WHEN SUM(CASE WHEN event_type = 'addtocart' THEN 1 ELSE 0 END) > 0 THEN 1
        ELSE 0
    END AS has_addtocart
FROM mart.retailrocket_interactions
GROUP BY user_id, item_id;

-- ---------------------------------------------------------------------
-- User feature mart
-- ---------------------------------------------------------------------

CREATE OR REPLACE TABLE mart.retailrocket_user_features AS
SELECT
    user_id,
    COUNT(*) AS total_events,
    COUNT(DISTINCT item_id) AS unique_items,
    SUM(CASE WHEN event_type = 'view' THEN 1 ELSE 0 END) AS view_events,
    SUM(CASE WHEN event_type = 'addtocart' THEN 1 ELSE 0 END) AS addtocart_events,
    SUM(CASE WHEN event_type = 'transaction' THEN 1 ELSE 0 END) AS transaction_events,
    SUM(interaction_weight) AS total_interaction_score,
    AVG(interaction_weight) AS avg_interaction_weight,
    MIN(event_timestamp_ms) AS first_event_timestamp_ms,
    MAX(event_timestamp_ms) AS last_event_timestamp_ms,
    MIN(event_datetime) AS first_event_datetime,
    MAX(event_datetime) AS last_event_datetime
FROM mart.retailrocket_interactions
GROUP BY user_id;

-- ---------------------------------------------------------------------
-- Item feature mart
-- ---------------------------------------------------------------------

CREATE OR REPLACE TABLE mart.retailrocket_item_features AS
WITH item_event_agg AS (
    SELECT
        item_id,
        COUNT(*) AS total_events,
        COUNT(DISTINCT user_id) AS unique_users,
        SUM(CASE WHEN event_type = 'view' THEN 1 ELSE 0 END) AS view_events,
        SUM(CASE WHEN event_type = 'addtocart' THEN 1 ELSE 0 END) AS addtocart_events,
        SUM(CASE WHEN event_type = 'transaction' THEN 1 ELSE 0 END) AS transaction_events,
        SUM(interaction_weight) AS total_interaction_score,
        AVG(interaction_weight) AS avg_interaction_weight,
        MIN(event_timestamp_ms) AS first_event_timestamp_ms,
        MAX(event_timestamp_ms) AS last_event_timestamp_ms,
        MIN(event_datetime) AS first_event_datetime,
        MAX(event_datetime) AS last_event_datetime
    FROM mart.retailrocket_interactions
    GROUP BY item_id
)
SELECT
    item.item_id,
    item.total_events,
    item.unique_users,
    item.view_events,
    item.addtocart_events,
    item.transaction_events,
    item.total_interaction_score,
    item.avg_interaction_weight,
    item.first_event_timestamp_ms,
    item.last_event_timestamp_ms,
    item.first_event_datetime,
    item.last_event_datetime,
    category.category_id,
    tree.parent_category_id,
    availability.is_available,
    availability.available_raw_value,
    CASE
        WHEN category.item_id IS NULL THEN 0
        ELSE 1
    END AS has_category_metadata,
    CASE
        WHEN availability.item_id IS NULL THEN 0
        ELSE 1
    END AS has_availability_metadata
FROM item_event_agg item
LEFT JOIN staged.retailrocket_item_category_latest category
    ON item.item_id = category.item_id
LEFT JOIN staged.retailrocket_category_tree tree
    ON category.category_id = tree.category_id
LEFT JOIN staged.retailrocket_item_availability_latest availability
    ON item.item_id = availability.item_id;