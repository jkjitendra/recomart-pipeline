-- Retailrocket DuckDB warehouse transformation.
-- Loads staged and curated Retailrocket Parquet files and creates recommender mart tables.

DROP SCHEMA IF EXISTS staged CASCADE;
DROP SCHEMA IF EXISTS curated CASCADE;
DROP SCHEMA IF EXISTS mart CASCADE;
DROP SCHEMA IF EXISTS features CASCADE;

CREATE SCHEMA IF NOT EXISTS staged;
CREATE SCHEMA IF NOT EXISTS curated;
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

CREATE OR REPLACE TABLE staged.retailrocket_item_properties_batch AS
SELECT *
FROM read_parquet('data/staged/source=retailrocket/item_properties_batch.parquet', hive_partitioning = false);

CREATE OR REPLACE TABLE staged.retailrocket_item_properties_api_delta AS
SELECT *
FROM read_parquet('data/staged/source=retailrocket/item_properties_api_delta.parquet', hive_partitioning = false);

CREATE OR REPLACE TABLE staged.retailrocket_item_properties_combined AS
SELECT *
FROM read_parquet('data/staged/source=retailrocket/item_properties_combined.parquet', hive_partitioning = false);

CREATE OR REPLACE TABLE staged.retailrocket_item_category_latest AS
SELECT *
FROM read_parquet('data/staged/source=retailrocket/item_category_latest.parquet', hive_partitioning = false);

CREATE OR REPLACE TABLE staged.retailrocket_item_availability_latest AS
SELECT *
FROM read_parquet('data/staged/source=retailrocket/item_availability_latest.parquet', hive_partitioning = false);

-- ---------------------------------------------------------------------
-- Curated tables
-- ---------------------------------------------------------------------

CREATE OR REPLACE TABLE curated.retailrocket_interactions AS
SELECT *
FROM read_parquet('data/curated/source=retailrocket/curated_interactions.parquet', hive_partitioning = false);

CREATE OR REPLACE TABLE curated.retailrocket_items AS
SELECT *
FROM read_parquet('data/curated/source=retailrocket/curated_items.parquet', hive_partitioning = false);

CREATE OR REPLACE TABLE curated.retailrocket_user_item_interactions AS
SELECT *
FROM read_parquet('data/curated/source=retailrocket/curated_user_item_interactions.parquet', hive_partitioning = false);

-- ---------------------------------------------------------------------
-- Mart views and tables
-- ---------------------------------------------------------------------

CREATE OR REPLACE VIEW mart.retailrocket_interactions AS
SELECT
    user_id,
    item_id,
    event_type,
    event_timestamp_ms,
    event_datetime,
    transaction_id,
    interaction_weight,
    category_id,
    parent_category_id,
    is_available,
    available_raw_value,
    has_category_metadata,
    has_availability_metadata
FROM curated.retailrocket_interactions;

CREATE OR REPLACE TABLE mart.retailrocket_user_item_interactions AS
SELECT
    user_id,
    item_id,
    total_events,
    view_count,
    addtocart_count,
    transaction_count,
    interaction_score,
    max_event_weight,
    first_event_timestamp_ms,
    last_event_timestamp_ms,
    first_event_datetime,
    last_event_datetime,
    category_id,
    parent_category_id,
    is_available,
    has_category_metadata,
    has_availability_metadata,
    has_transaction,
    has_addtocart
FROM curated.retailrocket_user_item_interactions;

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

CREATE OR REPLACE TABLE mart.retailrocket_item_features AS
SELECT
    item_id,
    total_events,
    unique_users,
    view_events,
    addtocart_events,
    transaction_events,
    (
        view_events * 1.0
        + addtocart_events * 3.0
        + transaction_events * 5.0
    ) AS total_interaction_score,
    CASE
        WHEN total_events > 0 THEN (
            view_events * 1.0
            + addtocart_events * 3.0
            + transaction_events * 5.0
        ) / total_events
        ELSE 0.0
    END AS avg_interaction_weight,
    first_event_timestamp_ms,
    last_event_timestamp_ms,
    first_event_datetime,
    last_event_datetime,
    category_id,
    parent_category_id,
    is_available,
    available_raw_value,
    has_category_metadata,
    has_availability_metadata
FROM curated.retailrocket_items;
