# RecoMart Retailrocket Recommendation Pipeline

RecoMart is a Retailrocket-only end-to-end data management pipeline for recommendation modeling. The implementation ingests batch CSV interaction data and REST/mock API item metadata deltas, stores raw snapshots in a partitioned local data lake, validates and prepares data, loads a DuckDB warehouse, builds feature tables and a custom feature registry, trains Retailrocket recommendation models, tracks runs with MLflow, orchestrates execution with Prefect, and generates reproducible assignment evidence.

## 1. Problem Formulation
The business goal is to recommend relevant products to Retailrocket users from interaction history and catalog metadata. The pipeline produces validated datasets, feature tables, model artifacts, inference outputs, lineage metadata, and a consolidated report suitable for submission.

- Recommendation target: rank candidate items for users.
- Main model: Retailrocket content-based recommender.
- Baseline model: event-weighted Retailrocket popularity recommender.
- Metrics: HitRate@10, Precision@10, Recall@10, and NDCG@10.

## 2. Data Sources
RecoMart uses Retailrocket data only.

- Batch CSV source: `events.csv`, `item_properties_part1.csv`, and `category_tree.csv`.
- REST API source: teammate-hosted item property delta endpoint.
- Mock API fallback: reproducible local mode that uses deterministic sample records with cursor state.
- API default batch size: 50 records per run; real API mode sends `count=<page_size>` and the server returns non-repeated rows.

### 2.1 External Source Summary
| dataset_name          | file_path                                            |   file_size_mb |     row_count |   distinct_visitors |   distinct_items |   distinct_properties |   distinct_categories |   distinct_parent_categories |   root_category_rows |   rows_with_transaction_id |   rows_without_transaction_id |   null_value_rows |   min_timestamp_ms |   max_timestamp_ms | min_timestamp_datetime     | max_timestamp_datetime     |
|:----------------------|:-----------------------------------------------------|---------------:|--------------:|--------------------:|-----------------:|----------------------:|----------------------:|-----------------------------:|---------------------:|---------------------------:|------------------------------:|------------------:|-------------------:|-------------------:|:---------------------------|:---------------------------|
| events                | data/external/retailrocket/events.csv                |         89.872 |    2.7561e+06 |         1.40758e+06 |           235061 |                   nan |                   nan |                          nan |                  nan |                      22457 |                   2.73364e+06 |               nan |        1.43062e+12 |        1.44255e+12 | 2015-05-03T03:00:04.384000 | 2015-09-18T02:59:47.788000 |
| item_properties_part1 | data/external/retailrocket/item_properties_part1.csv |        461.879 |    1.1e+07    |       nan           |           417053 |                  1097 |                   nan |                          nan |                  nan |                        nan |                 nan           |                 0 |        1.43123e+12 |        1.44211e+12 | 2015-05-10T03:00:00        | 2015-09-13T03:00:00        |
| category_tree         | data/external/retailrocket/category_tree.csv         |          0.014 | 1669          |       nan           |              nan |                   nan |                  1669 |                          362 |                   25 |                        nan |                 nan           |               nan |      nan           |      nan           | nan                        | nan                        |

### 2.2 Event Distribution
| event       |   row_count |   distinct_visitors |   distinct_items |   rows_with_transaction_id |
|:------------|------------:|--------------------:|-----------------:|---------------------------:|
| view        |     2664312 |             1404179 |           234838 |                          0 |
| addtocart   |       69332 |               37722 |            23903 |                          0 |
| transaction |       22457 |               11719 |            12025 |                      22457 |

## 3. Batch and REST/Mock API Ingestion
Batch ingestion copies the Retailrocket CSV source files from `data/external/retailrocket/` into immutable raw snapshots. API ingestion fetches item property delta records from a configurable teammate-hosted REST endpoint or the local mock mode used for reproducible assignment runs. The real API contract returns `data`, `success`, `count`, `total_rows`, and `unread_rows`; the mock contract returns `records`, `next_cursor`, and `has_more`. Both contracts are normalized to the same raw schema.

API environment variables:

```text
RECOMART_CATALOG_API_BASE_URL
RECOMART_CATALOG_API_ENDPOINT
RECOMART_CATALOG_API_TIMEOUT_SEC
RECOMART_CATALOG_API_PAGE_SIZE
RECOMART_CATALOG_API_AUTH_TOKEN
RECOMART_CATALOG_API_STATE_FILE
RECOMART_CATALOG_API_MOCK_MODE
```

Default real API command:

```bash
python -m src.ingestion.ingest_retailrocket_catalog_api
```

Mock fallback command:

```bash
RECOMART_CATALOG_API_MOCK_MODE=true python -m src.ingestion.ingest_retailrocket_catalog_api
```

Real API mode command:

```bash
RECOMART_CATALOG_API_MOCK_MODE=false \
RECOMART_CATALOG_API_BASE_URL="https://recomart-flask.295uyonmxxer.us-south.codeengine.appdomain.cloud" \
RECOMART_CATALOG_API_ENDPOINT="/items" \
RECOMART_CATALOG_API_PAGE_SIZE=50 \
python -m src.ingestion.ingest_retailrocket_catalog_api
```

The default runtime configuration uses the teammate-hosted endpoint `https://recomart-flask.295uyonmxxer.us-south.codeengine.appdomain.cloud/items` with `count=<RECOMART_CATALOG_API_PAGE_SIZE>`. The API serves new rows without repetition.

## 4. Raw Data Lake Layout
Raw data uses one timestamp partition field:

```text
data/raw/source=<source_name>/type=<data_type>/ingestion_timestamp=<YYYYMMDD_HHMMSS>/
```

Implemented raw snapshot paths:

```text
data/raw/source=retailrocket_batch/type=events/ingestion_timestamp=<YYYYMMDD_HHMMSS>/
data/raw/source=retailrocket_batch/type=item_properties_part1/ingestion_timestamp=<YYYYMMDD_HHMMSS>/
data/raw/source=retailrocket_batch/type=category_tree/ingestion_timestamp=<YYYYMMDD_HHMMSS>/
data/raw/source=retailrocket_api/type=item_properties_delta/ingestion_timestamp=<YYYYMMDD_HHMMSS>/
```

## 5. Data Validation Summary
### 5.1 Raw Validation
| status   |   check_count |
|:---------|--------------:|
| PASS     |            91 |

### 5.2 Staged and Curated Validation
| status   |   check_count |
|:---------|--------------:|
| PASS     |            59 |
| WARN     |             3 |

Validation checks cover file presence, schema, row counts, nulls, duplicates, event type validity, timestamp validity, API delta presence, latest metadata uniqueness, and referential coverage between interactions and metadata. The current reports show no failing validation checks.

Warning-level checks are documented and do not block the pipeline. Current warnings are related to duplicate event groups and item metadata coverage for category and availability fields:

| dataset_name         | check_name                    | status   | details                     |
|:---------------------|:------------------------------|:---------|:----------------------------|
| events               | duplicate_event_rows          | WARN     | Duplicate event groups: 458 |
| referential_coverage | event_items_with_category     | WARN     | 102390/235061 (0.4356)      |
| referential_coverage | event_items_with_availability | WARN     | 108603/235061 (0.4620)      |

## 6. Preparation, Staged Layer, and Curated Layer
Preparation reads from `data/raw/`, uses latest batch partitions for batch source types, and reads all API delta partitions so metadata deltas accumulate across runs. Staged outputs are typed Parquet files; curated outputs are EDA/modeling-ready analytical datasets.

### 6.1 Staged Preparation Summary
| dataset_name              | output_file                                                       | description                                                                   | source_files                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |   row_count |   column_count |   file_size_mb | status   |   api_delta_partition_count |   api_delta_raw_rows |   api_delta_rows_after_deduplication | api_delta_deduplication_key             |
|:--------------------------|:------------------------------------------------------------------|:------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------:|---------------:|---------------:|:---------|----------------------------:|---------------------:|-------------------------------------:|:----------------------------------------|
| events                    | data/staged/source=retailrocket/events.parquet                    | Retailrocket visitor-item events prepared from raw batch ingestion.           | data/raw/source=retailrocket_batch/type=events/ingestion_timestamp=20260428_224249/events.csv                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |     2756101 |              8 |         54.964 | PASS     |                         nan |                  nan |                                  nan | nan                                     |
| category_tree             | data/staged/source=retailrocket/category_tree.parquet             | Retailrocket category hierarchy prepared from raw batch ingestion.            | data/raw/source=retailrocket_batch/type=category_tree/ingestion_timestamp=20260428_224249/category_tree.csv                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |        1669 |              5 |          0.012 | PASS     |                         nan |                  nan |                                  nan | nan                                     |
| item_properties_batch     | data/staged/source=retailrocket/item_properties_batch.parquet     | Selected Retailrocket batch item properties from raw item_properties_part1.   | data/raw/source=retailrocket_batch/type=item_properties_part1/ingestion_timestamp=20260428_224249/item_properties_part1.csv                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |     1243692 |              8 |          8.857 | PASS     |                         nan |                  nan |                                  nan | nan                                     |
| item_properties_api_delta | data/staged/source=retailrocket/item_properties_api_delta.parquet | Retailrocket item property delta records from all REST/mock API raw landings. | data/raw/source=retailrocket_api/type=item_properties_delta/ingestion_timestamp=20260426_230121/item_properties_delta.parquet; data/raw/source=retailrocket_api/type=item_properties_delta/ingestion_timestamp=20260426_230542/item_properties_delta.parquet; data/raw/source=retailrocket_api/type=item_properties_delta/ingestion_timestamp=20260427_180102/item_properties_delta.parquet; data/raw/source=retailrocket_api/type=item_properties_delta/ingestion_timestamp=20260427_180548/item_properties_delta.parquet; data/raw/source=retailrocket_api/type=item_properties_delta/ingestion_timestamp=20260427_234351/item_properties_delta.parquet; data/raw/source=retailrocket_api/type=item_properties_delta/ingestion_timestamp=20260427_235803/item_properties_delta.parquet; data/raw/source=retailrocket_api/type=item_properties_delta/ingestion_timestamp=20260428_222649/item_properties_delta.parquet; data/raw/source=retailrocket_api/type=item_properties_delta/ingestion_timestamp=20260428_223935/item_properties_delta.parquet; data/raw/source=retailrocket_api/type=item_properties_delta/ingestion_timestamp=20260428_224313/item_properties_delta.parquet |         210 |              9 |          0.007 | PASS     |                           9 |                  210 |                                  210 | none - all API delta rows are preserved |
| item_properties_combined  | data/staged/source=retailrocket/item_properties_combined.parquet  | Combined batch and API delta item properties.                                 | nan                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |     1243902 |              9 |          8.864 | PASS     |                         nan |                  nan |                                  nan | nan                                     |
| item_category_latest      | data/staged/source=retailrocket/item_category_latest.parquet      | Latest categoryid value per item from combined item properties.               | nan                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |      229346 |              6 |          1.5   | PASS     |                         nan |                  nan |                                  nan | nan                                     |
| item_availability_latest  | data/staged/source=retailrocket/item_availability_latest.parquet  | Latest availability value per item from combined item properties.             | nan                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |      236885 |              7 |          1.295 | PASS     |                         nan |                  nan |                                  nan | nan                                     |

### 6.2 Curated Dataset Summary
| dataset_name                   | output_file                                                             | description                                                                 |   source_files |   row_count |   column_count |   file_size_mb | status   |
|:-------------------------------|:------------------------------------------------------------------------|:----------------------------------------------------------------------------|---------------:|------------:|---------------:|---------------:|:---------|
| curated_interactions           | data/curated/source=retailrocket/curated_interactions.parquet           | EDA-ready Retailrocket interaction rows enriched with latest item metadata. |            nan |     2756101 |             13 |         60.751 | PASS     |
| curated_items                  | data/curated/source=retailrocket/curated_items.parquet                  | EDA-ready Retailrocket item dataset with event aggregates and metadata.     |            nan |      235061 |             16 |          8.487 | PASS     |
| curated_user_item_interactions | data/curated/source=retailrocket/curated_user_item_interactions.parquet | EDA/modeling-ready user-item interaction aggregates.                        |            nan |     2145179 |             19 |         80.775 | PASS     |

## 7. EDA Plots
Generated EDA/model evidence plots are stored under `reports/plots/`:

- `retailrocket_event_distribution.png`
- `retailrocket_top_items.png`
- `retailrocket_user_activity_distribution.png`
- `retailrocket_model_metrics.png`
- `retailrocket_model_comparison_metrics.png`

## 8. SQL Transformation and Warehouse
DuckDB stores Retailrocket staged, curated, mart, and feature schemas in `data/warehouse/recomart.duckdb`. The DuckDB binary is generated locally and is not committed directly to Git.

### 8.1 DuckDB Load Summary
| schema_name   | relation_name                          | relation_type   |   row_count |   column_count | status   |
|:--------------|:---------------------------------------|:----------------|------------:|---------------:|:---------|
| staged        | retailrocket_events                    | BASE TABLE      |     2756101 |              8 | PASS     |
| staged        | retailrocket_category_tree             | BASE TABLE      |        1669 |              5 | PASS     |
| staged        | retailrocket_item_properties_batch     | BASE TABLE      |     1243692 |              8 | PASS     |
| staged        | retailrocket_item_properties_api_delta | BASE TABLE      |         210 |              9 | PASS     |
| staged        | retailrocket_item_properties_combined  | BASE TABLE      |     1243902 |              9 | PASS     |
| staged        | retailrocket_item_category_latest      | BASE TABLE      |      229346 |              6 | PASS     |
| staged        | retailrocket_item_availability_latest  | BASE TABLE      |      236885 |              7 | PASS     |
| curated       | retailrocket_interactions              | BASE TABLE      |     2756101 |             13 | PASS     |
| curated       | retailrocket_items                     | BASE TABLE      |      235061 |             16 | PASS     |
| curated       | retailrocket_user_item_interactions    | BASE TABLE      |     2145179 |             19 | PASS     |
| mart          | retailrocket_interactions              | VIEW            |     2756101 |             13 | PASS     |
| mart          | retailrocket_user_item_interactions    | BASE TABLE      |     2145179 |             19 | PASS     |
| mart          | retailrocket_user_features             | BASE TABLE      |     1407580 |             12 | PASS     |
| mart          | retailrocket_item_features             | BASE TABLE      |      235061 |             18 | PASS     |

### 8.2 SQL Schema Sample
| schema_name   | table_name                | column_name               |   ordinal_position | data_type                | is_nullable   |
|:--------------|:--------------------------|:--------------------------|-------------------:|:-------------------------|:--------------|
| curated       | retailrocket_interactions | user_id                   |                  1 | BIGINT                   | YES           |
| curated       | retailrocket_interactions | item_id                   |                  2 | BIGINT                   | YES           |
| curated       | retailrocket_interactions | event_type                |                  3 | VARCHAR                  | YES           |
| curated       | retailrocket_interactions | event_timestamp_ms        |                  4 | BIGINT                   | YES           |
| curated       | retailrocket_interactions | event_datetime            |                  5 | TIMESTAMP WITH TIME ZONE | YES           |
| curated       | retailrocket_interactions | transaction_id            |                  6 | BIGINT                   | YES           |
| curated       | retailrocket_interactions | interaction_weight        |                  7 | DECIMAL(2,1)             | YES           |
| curated       | retailrocket_interactions | category_id               |                  8 | BIGINT                   | YES           |
| curated       | retailrocket_interactions | parent_category_id        |                  9 | BIGINT                   | YES           |
| curated       | retailrocket_interactions | is_available              |                 10 | INTEGER                  | YES           |
| curated       | retailrocket_interactions | available_raw_value       |                 11 | VARCHAR                  | YES           |
| curated       | retailrocket_interactions | has_category_metadata     |                 12 | INTEGER                  | YES           |
| curated       | retailrocket_interactions | has_availability_metadata |                 13 | INTEGER                  | YES           |
| curated       | retailrocket_items        | item_id                   |                  1 | BIGINT                   | YES           |
| curated       | retailrocket_items        | total_events              |                  2 | BIGINT                   | YES           |
| curated       | retailrocket_items        | unique_users              |                  3 | BIGINT                   | YES           |
| curated       | retailrocket_items        | view_events               |                  4 | DOUBLE                   | YES           |
| curated       | retailrocket_items        | addtocart_events          |                  5 | DOUBLE                   | YES           |
| curated       | retailrocket_items        | transaction_events        |                  6 | DOUBLE                   | YES           |
| curated       | retailrocket_items        | first_event_timestamp_ms  |                  7 | BIGINT                   | YES           |
| curated       | retailrocket_items        | last_event_timestamp_ms   |                  8 | BIGINT                   | YES           |
| curated       | retailrocket_items        | first_event_datetime      |                  9 | TIMESTAMP WITH TIME ZONE | YES           |
| curated       | retailrocket_items        | last_event_datetime       |                 10 | TIMESTAMP WITH TIME ZONE | YES           |
| curated       | retailrocket_items        | category_id               |                 11 | BIGINT                   | YES           |
| curated       | retailrocket_items        | parent_category_id        |                 12 | BIGINT                   | YES           |

## 9. Feature Engineering
Feature tables are generated under `data/features/source=retailrocket/`.

### 9.1 Feature Output Summary
| schema_name   | table_name                      | output_file                                                  |   row_count |   column_count |   file_size_mb | status   |
|:--------------|:--------------------------------|:-------------------------------------------------------------|------------:|---------------:|---------------:|:---------|
| features      | retailrocket_user_features      | data/features/source=retailrocket/user_features.parquet      |     1407580 |             16 |         47.358 | PASS     |
| features      | retailrocket_item_features      | data/features/source=retailrocket/item_features.parquet      |      235061 |             21 |          8.988 | PASS     |
| features      | retailrocket_user_item_features | data/features/source=retailrocket/user_item_features.parquet |     2145179 |             24 |         81.492 | PASS     |

### 9.2 Feature Logic Summary
| feature_group                   | feature_or_group                                           | source                                                                      | logic                                                                                                    | used_for                                                   |
|:--------------------------------|:-----------------------------------------------------------|:----------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------|
| retailrocket_user_features      | total_events                                               | mart.retailrocket_interactions                                              | Count all events generated by each user.                                                                 | User activity representation, training, inference analysis |
| retailrocket_user_features      | unique_items                                               | mart.retailrocket_interactions                                              | Count distinct items interacted with by each user.                                                       | User diversity and activity profiling                      |
| retailrocket_user_features      | view_events, addtocart_events, transaction_events          | mart.retailrocket_interactions                                              | Conditional event counts by user and event type.                                                         | Intent-strength measurement                                |
| retailrocket_user_features      | addtocart_event_rate, transaction_event_rate               | features.retailrocket_user_features                                         | Event-type count divided by total user event count.                                                      | User conversion tendency                                   |
| retailrocket_item_features      | total_events, unique_users                                 | mart.retailrocket_interactions                                              | Count total item interactions and distinct users per item.                                               | Item popularity and candidate ranking                      |
| retailrocket_item_features      | category_id, parent_category_id                            | staged.retailrocket_item_category_latest, staged.retailrocket_category_tree | Attach latest item category and parent category metadata.                                                | Content-aware filtering and catalog analysis               |
| retailrocket_item_features      | is_available                                               | staged.retailrocket_item_availability_latest                                | Attach latest availability flag for each item.                                                           | Candidate filtering and item metadata enrichment           |
| retailrocket_item_features      | addtocart_rate, transaction_rate, cart_to_transaction_rate | features.retailrocket_item_features                                         | Conversion ratios derived from item-level event counts.                                                  | Ranking and item quality signals                           |
| retailrocket_user_item_features | view_count, addtocart_count, transaction_count             | mart.retailrocket_user_item_interactions                                    | Aggregate user-item event counts by event type.                                                          | Implicit feedback model training                           |
| retailrocket_user_item_features | implicit_label                                             | features.retailrocket_user_item_features                                    | Encode interaction strength: view < addtocart < transaction.                                             | Recommendation target signal                               |
| retailrocket_user_item_features | strongest_event_type                                       | features.retailrocket_user_item_features                                    | Select strongest observed user-item event type.                                                          | Interpretability and behavioral analysis                   |
| retailrocket_feature_registry   | user, item, user-item feature views                        | configs/feature_store/retailrocket_feature_registry.json                    | Custom registry maps Retailrocket feature views to entities, source paths, versions, and intended usage. | Feature store demonstration                                |

## 10. Feature Store Registry and Retrieval Demo
The custom Retailrocket feature registry documents feature views, entity keys, source paths, data types, feature roles, usage, and source transformations. The retrieval demo reads current feature parquet files and writes representative user, item, and user-item samples.

### 10.1 Feature Metadata Documentation Sample
| registry_name                                | registry_version   | feature_view_name          | feature_view_version   | entity_keys   | source_path                                             | source_table                        | source_exists   | column_name              | data_type             | feature_role   | transformation                                                                                                                                  | used_for                                            | created_by                    | description                                                                     |
|:---------------------------------------------|:-------------------|:---------------------------|:-----------------------|:--------------|:--------------------------------------------------------|:------------------------------------|:----------------|:-------------------------|:----------------------|:---------------|:------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------|:------------------------------|:--------------------------------------------------------------------------------|
| retailrocket_recommendation_feature_registry | v1                 | retailrocket_user_features | v1                     | ["user_id"]   | data/features/source=retailrocket/user_features.parquet | features.retailrocket_user_features | True            | user_id                  | int64                 | entity_key     | Aggregates Retailrocket user interaction history from mart.retailrocket_user_features and derives activity, conversion, and intensity features. | training, inference, profiling                      | sql/retailrocket_features.sql | User-level behavioral features for Retailrocket visitors.                       |
| retailrocket_recommendation_feature_registry | v1                 | retailrocket_user_features | v1                     | ["user_id"]   | data/features/source=retailrocket/user_features.parquet | features.retailrocket_user_features | True            | total_events             | int64                 | feature        | Aggregates Retailrocket user interaction history from mart.retailrocket_user_features and derives activity, conversion, and intensity features. | training, inference, profiling                      | sql/retailrocket_features.sql | User-level behavioral features for Retailrocket visitors.                       |
| retailrocket_recommendation_feature_registry | v1                 | retailrocket_user_features | v1                     | ["user_id"]   | data/features/source=retailrocket/user_features.parquet | features.retailrocket_user_features | True            | unique_items             | int64                 | feature        | Aggregates Retailrocket user interaction history from mart.retailrocket_user_features and derives activity, conversion, and intensity features. | training, inference, profiling                      | sql/retailrocket_features.sql | User-level behavioral features for Retailrocket visitors.                       |
| retailrocket_recommendation_feature_registry | v1                 | retailrocket_user_features | v1                     | ["user_id"]   | data/features/source=retailrocket/user_features.parquet | features.retailrocket_user_features | True            | view_events              | double                | feature        | Aggregates Retailrocket user interaction history from mart.retailrocket_user_features and derives activity, conversion, and intensity features. | training, inference, profiling                      | sql/retailrocket_features.sql | User-level behavioral features for Retailrocket visitors.                       |
| retailrocket_recommendation_feature_registry | v1                 | retailrocket_user_features | v1                     | ["user_id"]   | data/features/source=retailrocket/user_features.parquet | features.retailrocket_user_features | True            | addtocart_events         | double                | feature        | Aggregates Retailrocket user interaction history from mart.retailrocket_user_features and derives activity, conversion, and intensity features. | training, inference, profiling                      | sql/retailrocket_features.sql | User-level behavioral features for Retailrocket visitors.                       |
| retailrocket_recommendation_feature_registry | v1                 | retailrocket_user_features | v1                     | ["user_id"]   | data/features/source=retailrocket/user_features.parquet | features.retailrocket_user_features | True            | transaction_events       | double                | feature        | Aggregates Retailrocket user interaction history from mart.retailrocket_user_features and derives activity, conversion, and intensity features. | training, inference, profiling                      | sql/retailrocket_features.sql | User-level behavioral features for Retailrocket visitors.                       |
| retailrocket_recommendation_feature_registry | v1                 | retailrocket_user_features | v1                     | ["user_id"]   | data/features/source=retailrocket/user_features.parquet | features.retailrocket_user_features | True            | total_interaction_score  | decimal128(38, 1)     | feature        | Aggregates Retailrocket user interaction history from mart.retailrocket_user_features and derives activity, conversion, and intensity features. | training, inference, profiling                      | sql/retailrocket_features.sql | User-level behavioral features for Retailrocket visitors.                       |
| retailrocket_recommendation_feature_registry | v1                 | retailrocket_user_features | v1                     | ["user_id"]   | data/features/source=retailrocket/user_features.parquet | features.retailrocket_user_features | True            | avg_interaction_weight   | double                | feature        | Aggregates Retailrocket user interaction history from mart.retailrocket_user_features and derives activity, conversion, and intensity features. | training, inference, profiling                      | sql/retailrocket_features.sql | User-level behavioral features for Retailrocket visitors.                       |
| retailrocket_recommendation_feature_registry | v1                 | retailrocket_user_features | v1                     | ["user_id"]   | data/features/source=retailrocket/user_features.parquet | features.retailrocket_user_features | True            | first_event_timestamp_ms | int64                 | metadata       | Aggregates Retailrocket user interaction history from mart.retailrocket_user_features and derives activity, conversion, and intensity features. | training, inference, profiling                      | sql/retailrocket_features.sql | User-level behavioral features for Retailrocket visitors.                       |
| retailrocket_recommendation_feature_registry | v1                 | retailrocket_user_features | v1                     | ["user_id"]   | data/features/source=retailrocket/user_features.parquet | features.retailrocket_user_features | True            | last_event_timestamp_ms  | int64                 | metadata       | Aggregates Retailrocket user interaction history from mart.retailrocket_user_features and derives activity, conversion, and intensity features. | training, inference, profiling                      | sql/retailrocket_features.sql | User-level behavioral features for Retailrocket visitors.                       |
| retailrocket_recommendation_feature_registry | v1                 | retailrocket_user_features | v1                     | ["user_id"]   | data/features/source=retailrocket/user_features.parquet | features.retailrocket_user_features | True            | first_event_datetime     | timestamp[us, tz=UTC] | metadata       | Aggregates Retailrocket user interaction history from mart.retailrocket_user_features and derives activity, conversion, and intensity features. | training, inference, profiling                      | sql/retailrocket_features.sql | User-level behavioral features for Retailrocket visitors.                       |
| retailrocket_recommendation_feature_registry | v1                 | retailrocket_user_features | v1                     | ["user_id"]   | data/features/source=retailrocket/user_features.parquet | features.retailrocket_user_features | True            | last_event_datetime      | timestamp[us, tz=UTC] | metadata       | Aggregates Retailrocket user interaction history from mart.retailrocket_user_features and derives activity, conversion, and intensity features. | training, inference, profiling                      | sql/retailrocket_features.sql | User-level behavioral features for Retailrocket visitors.                       |
| retailrocket_recommendation_feature_registry | v1                 | retailrocket_user_features | v1                     | ["user_id"]   | data/features/source=retailrocket/user_features.parquet | features.retailrocket_user_features | True            | active_days              | int64                 | feature        | Aggregates Retailrocket user interaction history from mart.retailrocket_user_features and derives activity, conversion, and intensity features. | training, inference, profiling                      | sql/retailrocket_features.sql | User-level behavioral features for Retailrocket visitors.                       |
| retailrocket_recommendation_feature_registry | v1                 | retailrocket_user_features | v1                     | ["user_id"]   | data/features/source=retailrocket/user_features.parquet | features.retailrocket_user_features | True            | addtocart_event_rate     | double                | feature        | Aggregates Retailrocket user interaction history from mart.retailrocket_user_features and derives activity, conversion, and intensity features. | training, inference, profiling                      | sql/retailrocket_features.sql | User-level behavioral features for Retailrocket visitors.                       |
| retailrocket_recommendation_feature_registry | v1                 | retailrocket_user_features | v1                     | ["user_id"]   | data/features/source=retailrocket/user_features.parquet | features.retailrocket_user_features | True            | transaction_event_rate   | double                | feature        | Aggregates Retailrocket user interaction history from mart.retailrocket_user_features and derives activity, conversion, and intensity features. | training, inference, profiling                      | sql/retailrocket_features.sql | User-level behavioral features for Retailrocket visitors.                       |
| retailrocket_recommendation_feature_registry | v1                 | retailrocket_user_features | v1                     | ["user_id"]   | data/features/source=retailrocket/user_features.parquet | features.retailrocket_user_features | True            | events_per_unique_item   | double                | feature        | Aggregates Retailrocket user interaction history from mart.retailrocket_user_features and derives activity, conversion, and intensity features. | training, inference, profiling                      | sql/retailrocket_features.sql | User-level behavioral features for Retailrocket visitors.                       |
| retailrocket_recommendation_feature_registry | v1                 | retailrocket_item_features | v1                     | ["item_id"]   | data/features/source=retailrocket/item_features.parquet | features.retailrocket_item_features | True            | item_id                  | int64                 | entity_key     | Aggregates Retailrocket item interactions and joins latest category and availability metadata from mart.retailrocket_item_features.             | training, inference, profiling, candidate filtering | sql/retailrocket_features.sql | Item-level popularity, conversion, and metadata features for candidate ranking. |
| retailrocket_recommendation_feature_registry | v1                 | retailrocket_item_features | v1                     | ["item_id"]   | data/features/source=retailrocket/item_features.parquet | features.retailrocket_item_features | True            | total_events             | int64                 | feature        | Aggregates Retailrocket item interactions and joins latest category and availability metadata from mart.retailrocket_item_features.             | training, inference, profiling, candidate filtering | sql/retailrocket_features.sql | Item-level popularity, conversion, and metadata features for candidate ranking. |
| retailrocket_recommendation_feature_registry | v1                 | retailrocket_item_features | v1                     | ["item_id"]   | data/features/source=retailrocket/item_features.parquet | features.retailrocket_item_features | True            | unique_users             | int64                 | feature        | Aggregates Retailrocket item interactions and joins latest category and availability metadata from mart.retailrocket_item_features.             | training, inference, profiling, candidate filtering | sql/retailrocket_features.sql | Item-level popularity, conversion, and metadata features for candidate ranking. |
| retailrocket_recommendation_feature_registry | v1                 | retailrocket_item_features | v1                     | ["item_id"]   | data/features/source=retailrocket/item_features.parquet | features.retailrocket_item_features | True            | view_events              | double                | feature        | Aggregates Retailrocket item interactions and joins latest category and availability metadata from mart.retailrocket_item_features.             | training, inference, profiling, candidate filtering | sql/retailrocket_features.sql | Item-level popularity, conversion, and metadata features for candidate ranking. |

### 10.2 Feature Retrieval Demo
| retrieval_type     | feature_view_name               | feature_view_version   | entity_filter                  |   rows_returned |   columns_returned | sample_values_json                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
|:-------------------|:--------------------------------|:-----------------------|:-------------------------------|----------------:|-------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| user_features      | retailrocket_user_features      | v1                     | user_id=1369328                |               1 |                 16 | {"active_days": "41", "addtocart_event_rate": "0.004484304932735426", "addtocart_events": "1.0", "avg_interaction_weight": "1.0269058295964126", "events_per_unique_item": "37.166666666666664", "first_event_datetime": "2015-08-07 02:46:43.261000+00:00", "first_event_timestamp_ms": "1438915603261", "last_event_datetime": "2015-09-16 00:46:05.298000+00:00", "last_event_timestamp_ms": "1442364365298", "total_events": "223", "total_interaction_score": "229.0", "transaction_event_rate": "0.004484304932735426", "transaction_events": "1.0", "unique_items": "6", "user_id": "1369328", "view_events": "221.0"}                                                                                                                                                                                                                  |
| item_features      | retailrocket_item_features      | v1                     | item_id=356339                 |               1 |                 21 | {"addtocart_events": "1.0", "addtocart_rate": "0.004545454545454545", "available_raw_value": "1", "avg_interaction_weight": "1.027027027027027", "cart_to_transaction_rate": "1.0", "category_id": "1035.0", "first_event_datetime": "2015-08-09 03:23:02.140000+00:00", "first_event_timestamp_ms": "1439090582140", "has_availability_metadata": "1", "has_category_metadata": "1", "is_available": "1.0", "item_id": "356339", "last_event_datetime": "2015-09-16 00:46:05.298000+00:00", "last_event_timestamp_ms": "1442364365298", "parent_category_id": "920.0", "total_events": "222", "total_interaction_score": "228.0", "transaction_events": "1.0", "transaction_rate": "0.004545454545454545", "unique_users": "6", "view_events": "220.0"}                                                                                       |
| user_item_features | retailrocket_user_item_features | v1                     | user_id=1369328,item_id=356339 |               1 |                 24 | {"addtocart_count": "1.0", "category_id": "1035.0", "first_event_datetime": "2015-08-09 03:23:02.140000+00:00", "first_event_timestamp_ms": "1439090582140", "has_addtocart": "1", "has_availability_metadata": "1", "has_category_metadata": "1", "has_transaction": "1", "implicit_label": "5.0", "interaction_score": "220.0", "is_available": "1.0", "item_id": "356339", "last_event_datetime": "2015-09-16 00:46:05.298000+00:00", "last_event_timestamp_ms": "1442364365298", "max_event_weight": "5.0", "parent_category_id": "920.0", "strongest_event_type": "transaction", "total_events": "214", "transaction_count": "1.0", "user_id": "1369328", "user_item_addtocart_rate": "0.0047169811320754715", "user_item_cart_to_transaction_rate": "1.0", "user_item_transaction_rate": "0.0047169811320754715", "view_count": "212.0"} |

## 11. DVC Versioning Workflow
DVC tracks external data, raw snapshots, staged data, curated data, feature tables, and model artifacts. Git stores source code, lightweight reports, plots, and `.dvc` metadata.

| pipeline_layer   | description                                         | dvc_file                               | tracked_path              | md5                                  |   size_bytes |   nfiles | versioning_tool   | tracked_in_git   |
|:-----------------|:----------------------------------------------------|:---------------------------------------|:--------------------------|:-------------------------------------|-------------:|---------:|:------------------|:-----------------|
| external_data    | Retailrocket original external CSV dataset          | data/external/retailrocket.dvc         | retailrocket              | a9cdd04c0b079c0a7becea06b1f4c684.dir |    578568116 |        3 | DVC               | True             |
| raw_batch_data   | Retailrocket raw batch ingestion snapshots          | data/raw/source=retailrocket_batch.dvc | source=retailrocket_batch | cf01e1dd3404797e1850ef06b9761760.dir |   2314279352 |       24 | DVC               | True             |
| raw_api_data     | Retailrocket raw REST API catalog delta snapshots   | data/raw/source=retailrocket_api.dvc   | source=retailrocket_api   | da17fa3054c65f9043d03e51b80b24a8.dir |       117279 |       41 | DVC               | True             |
| staged_data      | Retailrocket staged Parquet data                    | data/staged/source=retailrocket.dvc    | source=retailrocket       | d269c649a76843cbb8081fbd39b88d7a.dir |     79161039 |        7 | DVC               | True             |
| curated_data     | Retailrocket curated analytical Parquet data        | data/curated/source=retailrocket.dvc   | source=retailrocket       | 3d593ee350f4f070ca353a8df1413453.dir |    157326798 |        3 | DVC               | True             |
| feature_data     | Retailrocket feature Parquet data                   | data/features/source=retailrocket.dvc  | source=retailrocket       | 3c75f7bc3979a697888faaa8ace22757.dir |    144581018 |        3 | DVC               | True             |
| model_artifact   | Retailrocket trained recommendation model artifacts | models/retailrocket.dvc                | retailrocket              | 2912611cc64ad5b0dce549382cd5e9fb.dir |      1226758 |        2 | DVC               | True             |

## 12. Model Training and Evaluation
### 12.1 Popularity Baseline
| model_name                          | model_type                       |   top_k |   test_window_days |   evaluation_sample_users |   min_timestamp_ms |   max_timestamp_ms |   cutoff_timestamp_ms | min_event_datetime               | max_event_datetime               |   train_event_count |   test_event_count |   train_distinct_items |   model_saved_item_count |   model_item_coverage |   test_target_users |   evaluated_users |   hits |   hit_rate_at_10 |   precision_at_10 |   recall_at_10 |   ndcg_at_10 |
|:------------------------------------|:---------------------------------|--------:|-------------------:|--------------------------:|-------------------:|-------------------:|----------------------:|:---------------------------------|:---------------------------------|--------------------:|-------------------:|-----------------------:|-------------------------:|----------------------:|--------------------:|------------------:|-------:|-----------------:|------------------:|---------------:|-------------:|
| retailrocket_popularity_recommender | event_weighted_global_popularity |      10 |                 14 |                     10000 |      1430622004384 |      1442545187788 |         1441335587788 | 2015-05-03 08:30:04.384000+05:30 | 2015-09-18 08:29:47.788000+05:30 |             2509138 |             246963 |                 225427 |                     5000 |             0.0221801 |                3648 |              3648 |     49 |         0.013432 |         0.0013432 |       0.013432 |   0.00762866 |

### 12.2 Content-Based Recommender
| model_name                             | model_type                       |   top_k |   test_window_days |   evaluation_sample_users |   min_timestamp_ms |   max_timestamp_ms |   cutoff_timestamp_ms |   candidate_item_count |   test_target_users |   user_profile_count |   evaluated_users |   hits |   hit_rate_at_10 |   precision_at_10 |   recall_at_10 |   ndcg_at_10 |
|:---------------------------------------|:---------------------------------|--------:|-------------------:|--------------------------:|-------------------:|-------------------:|----------------------:|-----------------------:|--------------------:|---------------------:|------------------:|-------:|-----------------:|------------------:|---------------:|-------------:|
| retailrocket_content_based_recommender | category_content_based_filtering |      10 |                 14 |                     10000 |      1430622004384 |      1442545187788 |         1441335587788 |                   5000 |                3648 |                  517 |              3648 |     27 |       0.00740132 |       0.000740132 |     0.00740132 |   0.00281525 |

### 12.3 Model Comparison
| model_family               | model_name                             | model_type                       |   top_k |   evaluated_users |   hits |   hit_rate_at_10 |   precision_at_10 |   recall_at_10 |   ndcg_at_10 | model_artifact_path                               | training_report_path                                    |
|:---------------------------|:---------------------------------------|:---------------------------------|--------:|------------------:|-------:|-----------------:|------------------:|---------------:|-------------:|:--------------------------------------------------|:--------------------------------------------------------|
| global_popularity_baseline | retailrocket_popularity_recommender    | event_weighted_global_popularity |      10 |              3648 |     49 |       0.013432   |       0.0013432   |     0.013432   |   0.00762866 | models/retailrocket/popularity_recommender.pkl    | reports/retailrocket_model_training_summary.csv         |
| content_based_filtering    | retailrocket_content_based_recommender | category_content_based_filtering |      10 |              3648 |     27 |       0.00740132 |       0.000740132 |     0.00740132 |   0.00281525 | models/retailrocket/content_based_recommender.pkl | reports/retailrocket_content_model_training_summary.csv |

## 13. MLflow Tracking Metadata
Both Retailrocket models log to the MLflow experiment `retailrocket_recommendation_models`. The run records include model parameters, HitRate@10, Precision@10, Recall@10, NDCG@10, and report/model artifacts. MLflow screenshot evidence is stored under `reports/screenshots/`.

## 14. Inference Demo
| model_name                             | model_type                       |   requested_user_count |   top_k |   recommendation_rows |   distinct_recommended_items |   training_hit_rate_at_10 |   training_precision_at_10 |   training_recall_at_10 |   test_target_users | model_key     | model_artifact_path                               |   candidate_or_saved_item_count |   training_ndcg_at_10 |   train_event_count |   test_event_count |
|:---------------------------------------|:---------------------------------|-----------------------:|--------:|----------------------:|-----------------------------:|--------------------------:|---------------------------:|------------------------:|--------------------:|:--------------|:--------------------------------------------------|--------------------------------:|----------------------:|--------------------:|-------------------:|
| retailrocket_popularity_recommender    | event_weighted_global_popularity |                      5 |       5 |                    25 |                           18 |                0.013432   |                0.0013432   |              0.013432   |                3648 | popularity    | models/retailrocket/popularity_recommender.pkl    |                            5000 |            0.00762866 |         2.50914e+06 |             246963 |
| retailrocket_content_based_recommender | category_content_based_filtering |                      5 |       5 |                    25 |                            7 |                0.00740132 |                0.000740132 |              0.00740132 |                3648 | content_based | models/retailrocket/content_based_recommender.pkl |                            5000 |            0.00281525 |       nan           |                nan |

## 15. Prefect Orchestration Summary
Prefect orchestrates the full local pipeline using command-based tasks. The summary records step order, step name, command, status, return code, timing, and stdout/stderr tails.

|   step_order | step_name                                 | status   |   return_code |   duration_seconds |
|-------------:|:------------------------------------------|:---------|--------------:|-------------------:|
|            1 | inspect_retailrocket_external             | PASS     |             0 |              4.734 |
|            2 | ingest_retailrocket_batch                 | PASS     |             0 |             10.774 |
|            3 | ingest_retailrocket_catalog_api_delta     | PASS     |             0 |              1.281 |
|            4 | validate_retailrocket_raw                 | PASS     |             0 |              9.987 |
|            5 | prepare_retailrocket_staged_and_curated   | PASS     |             0 |              6.056 |
|            6 | validate_retailrocket_staged_and_curated  | PASS     |             0 |              1.532 |
|            7 | load_retailrocket_to_duckdb               | PASS     |             0 |             11.573 |
|            8 | build_retailrocket_features               | PASS     |             0 |              7.511 |
|            9 | retrieve_retailrocket_features            | PASS     |             0 |              8.71  |
|           10 | train_retailrocket_popularity_recommender | PASS     |             0 |             40.412 |
|           11 | train_retailrocket_content_recommender    | PASS     |             0 |             73.57  |
|           12 | run_retailrocket_popularity_inference     | PASS     |             0 |              0.906 |
|           13 | run_retailrocket_content_based_inference  | PASS     |             0 |              0.965 |
|           14 | generate_model_comparison                 | PASS     |             0 |              1.273 |
|           15 | generate_assignment_evidence              | PASS     |             0 |              1.482 |
|           16 | generate_final_report                     | PASS     |             0 |              0.61  |
|           17 | generate_assignment_pdf                   | PASS     |             0 |              1.97  |

## 16. API Ingestion 30-Minute Schedule
The API-only Prefect flow can be deployed on a 30-minute interval with Prefect 3:

```bash
prefect deploy orchestration/retailrocket_api_ingestion_flow.py:retailrocket_api_ingestion_flow \
  --name retailrocket-api-every-30-minutes \
  --interval 1800 \
  --pool default-agent-pool
```

## 17. Known Validation Warnings
The current validation summary contains no failing checks. Warning-level checks are documented and do not block the pipeline. Current warnings, when present, are related to duplicate event groups and metadata coverage for category and availability fields.

| dataset_name         | check_name                    | status   | details                     |
|:---------------------|:------------------------------|:---------|:----------------------------|
| events               | duplicate_event_rows          | WARN     | Duplicate event groups: 458 |
| referential_coverage | event_items_with_category     | WARN     | 102390/235061 (0.4356)      |
| referential_coverage | event_items_with_availability | WARN     | 108603/235061 (0.4620)      |

## 18. Reproducibility Steps
```bash
conda activate recomart
dvc pull
python -m orchestration.retailrocket_pipeline
python -m src.reporting.generate_assignment_evidence
python -m src.reporting.generate_model_comparison
python -m src.reporting.generate_final_report
python -m src.reporting.generate_assignment_pdf
git status
dvc status
```

## 19. Conclusion
RecoMart implements the required data management pipeline stages for a recommendation system using Retailrocket batch data and a REST/mock API metadata feed. The repository contains source code, DVC metadata, validation and model reports, Prefect orchestration evidence, MLflow screenshot evidence, and the consolidated PDF report.