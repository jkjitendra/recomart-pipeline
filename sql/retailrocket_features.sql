-- Retailrocket feature engineering.
-- Builds feature tables from Retailrocket DuckDB mart tables.

CREATE SCHEMA IF NOT EXISTS features;

-- ---------------------------------------------------------------------
-- User features
-- ---------------------------------------------------------------------

CREATE OR REPLACE TABLE features.retailrocket_user_features AS
SELECT
    user_id,
    total_events,
    unique_items,
    view_events,
    addtocart_events,
    transaction_events,
    total_interaction_score,
    avg_interaction_weight,
    first_event_timestamp_ms,
    last_event_timestamp_ms,
    first_event_datetime,
    last_event_datetime,
    DATE_DIFF('day', CAST(first_event_datetime AS DATE), CAST(last_event_datetime AS DATE)) + 1 AS active_days,
    CASE
        WHEN total_events > 0 THEN CAST(addtocart_events AS DOUBLE) / total_events
        ELSE 0.0
    END AS addtocart_event_rate,
    CASE
        WHEN total_events > 0 THEN CAST(transaction_events AS DOUBLE) / total_events
        ELSE 0.0
    END AS transaction_event_rate,
    CASE
        WHEN unique_items > 0 THEN CAST(total_events AS DOUBLE) / unique_items
        ELSE 0.0
    END AS events_per_unique_item
FROM mart.retailrocket_user_features;

-- ---------------------------------------------------------------------
-- Item features
-- ---------------------------------------------------------------------

CREATE OR REPLACE TABLE features.retailrocket_item_features AS
SELECT
    item_id,
    total_events,
    unique_users,
    view_events,
    addtocart_events,
    transaction_events,
    total_interaction_score,
    avg_interaction_weight,
    first_event_timestamp_ms,
    last_event_timestamp_ms,
    first_event_datetime,
    last_event_datetime,
    category_id,
    parent_category_id,
    is_available,
    available_raw_value,
    has_category_metadata,
    has_availability_metadata,
    CASE
        WHEN view_events > 0 THEN CAST(addtocart_events AS DOUBLE) / view_events
        ELSE 0.0
    END AS addtocart_rate,
    CASE
        WHEN view_events > 0 THEN CAST(transaction_events AS DOUBLE) / view_events
        ELSE 0.0
    END AS transaction_rate,
    CASE
        WHEN addtocart_events > 0 THEN CAST(transaction_events AS DOUBLE) / addtocart_events
        ELSE 0.0
    END AS cart_to_transaction_rate
FROM mart.retailrocket_item_features;

-- ---------------------------------------------------------------------
-- User-item interaction features
-- ---------------------------------------------------------------------

CREATE OR REPLACE TABLE features.retailrocket_user_item_features AS
SELECT
    ui.user_id,
    ui.item_id,
    ui.total_events,
    ui.view_count,
    ui.addtocart_count,
    ui.transaction_count,
    ui.interaction_score,
    ui.max_event_weight,
    ui.first_event_timestamp_ms,
    ui.last_event_timestamp_ms,
    ui.first_event_datetime,
    ui.last_event_datetime,
    ui.has_transaction,
    ui.has_addtocart,
    item.category_id,
    item.parent_category_id,
    item.is_available,
    item.has_category_metadata,
    item.has_availability_metadata,
    CASE
        WHEN ui.view_count > 0 THEN CAST(ui.addtocart_count AS DOUBLE) / ui.view_count
        ELSE 0.0
    END AS user_item_addtocart_rate,
    CASE
        WHEN ui.view_count > 0 THEN CAST(ui.transaction_count AS DOUBLE) / ui.view_count
        ELSE 0.0
    END AS user_item_transaction_rate,
    CASE
        WHEN ui.addtocart_count > 0 THEN CAST(ui.transaction_count AS DOUBLE) / ui.addtocart_count
        ELSE 0.0
    END AS user_item_cart_to_transaction_rate,
    CASE
        WHEN ui.has_transaction = 1 THEN 5.0
        WHEN ui.has_addtocart = 1 THEN 3.0
        ELSE 1.0
    END AS implicit_label,
    CASE
        WHEN ui.has_transaction = 1 THEN 'transaction'
        WHEN ui.has_addtocart = 1 THEN 'addtocart'
        ELSE 'view'
    END AS strongest_event_type
FROM mart.retailrocket_user_item_interactions ui
LEFT JOIN features.retailrocket_item_features item
    ON ui.item_id = item.item_id;