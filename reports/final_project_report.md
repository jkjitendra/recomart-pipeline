# RecoMart: Data Management Pipeline for ML-Based Recommendation

This project implements an end-to-end data management and machine learning pipeline for recommendation systems. It covers ingestion, inspection, validation, staged storage, DuckDB warehousing, feature engineering, feature registry concepts, model training, MLflow tracking, DVC versioning, inference, and Prefect orchestration.

# 1. Project Objective
The objective is to build a reproducible ML data pipeline that converts raw recommendation data into validated features and trained recommender models. DummyJSON is used as a small API-based demo dataset, while Retailrocket is used as the main large-scale recommendation dataset.

# 2. Tools Used and Why
| Tool | Usage |
|---|---|
| Python | Main implementation language |
| pandas | Data inspection, summaries, reports |
| DuckDB | Local analytical warehouse and SQL transformation engine |
| Parquet | Efficient staged and feature storage format |
| DVC | Versioning large datasets and model artifacts |
| Git | Versioning source code and lightweight reports |
| MLflow | Experiment tracking and metric logging |
| Prefect | Pipeline orchestration |
| Docker | Available for containerized execution |


# 3. Repository Structure
```text
configs/          Feature registry configuration
data/external/    Retailrocket source data tracked by DVC
data/raw/         DummyJSON API data
data/staged/      Cleaned Parquet datasets tracked by DVC
data/features/    ML feature datasets tracked by DVC
data/warehouse/   DuckDB local warehouse
models/           Trained model artifacts tracked by DVC
reports/          CSV summaries, final report, screenshots
sql/              DuckDB SQL scripts
src/              Python modules
orchestration/    Prefect flows
```

# 4. Dataset Summary

## 4.1 DummyJSON Raw Summary
| file_path                                                                                    | file_name                     | top_level_keys               | list_key   |   record_count |
|:---------------------------------------------------------------------------------------------|:------------------------------|:-----------------------------|:-----------|---------------:|
| data/raw/source=dummyjson/type=carts/ingest_date=2026-04-20/carts_20260420_224802.json       | carts_20260420_224802.json    | carts, total, skip, limit    | carts      |            100 |
| data/raw/source=dummyjson/type=carts/ingest_date=2026-04-21/carts_20260421_031642.json       | carts_20260421_031642.json    | carts, total, skip, limit    | carts      |            100 |
| data/raw/source=dummyjson/type=carts/ingest_date=2026-04-22/carts_20260422_002638.json       | carts_20260422_002638.json    | carts, total, skip, limit    | carts      |            100 |
| data/raw/source=dummyjson/type=carts/ingest_date=2026-04-22/carts_20260422_020241.json       | carts_20260422_020241.json    | carts, total, skip, limit    | carts      |            100 |
| data/raw/source=dummyjson/type=products/ingest_date=2026-04-20/products_20260420_224800.json | products_20260420_224800.json | products, total, skip, limit | products   |            100 |
| data/raw/source=dummyjson/type=products/ingest_date=2026-04-21/products_20260421_031640.json | products_20260421_031640.json | products, total, skip, limit | products   |            100 |
| data/raw/source=dummyjson/type=products/ingest_date=2026-04-22/products_20260422_002636.json | products_20260422_002636.json | products, total, skip, limit | products   |            100 |
| data/raw/source=dummyjson/type=products/ingest_date=2026-04-22/products_20260422_020238.json | products_20260422_020238.json | products, total, skip, limit | products   |            100 |
| data/raw/source=dummyjson/type=users/ingest_date=2026-04-20/users_20260420_224801.json       | users_20260420_224801.json    | users, total, skip, limit    | users      |            100 |
| data/raw/source=dummyjson/type=users/ingest_date=2026-04-21/users_20260421_031641.json       | users_20260421_031641.json    | users, total, skip, limit    | users      |            100 |
| data/raw/source=dummyjson/type=users/ingest_date=2026-04-22/users_20260422_002637.json       | users_20260422_002637.json    | users, total, skip, limit    | users      |            100 |
| data/raw/source=dummyjson/type=users/ingest_date=2026-04-22/users_20260422_020240.json       | users_20260422_020240.json    | users, total, skip, limit    | users      |            100 |

## 4.2 Retailrocket External Summary
| dataset_name             | file_path                                                                                                  |   file_size_mb |      row_count |   distinct_visitors |   distinct_items |   distinct_properties |   distinct_categories |   distinct_parent_categories |   root_category_rows |   rows_with_transaction_id |   rows_without_transaction_id |   null_value_rows |   min_timestamp_ms |   max_timestamp_ms | min_timestamp_datetime     | max_timestamp_datetime     |
|:-------------------------|:-----------------------------------------------------------------------------------------------------------|---------------:|---------------:|--------------------:|-----------------:|----------------------:|----------------------:|-----------------------------:|---------------------:|---------------------------:|------------------------------:|------------------:|-------------------:|-------------------:|:---------------------------|:---------------------------|
| events                   | data/external/retailrocket/events.csv                                                                      |         89.872 |    2.7561e+06  |         1.40758e+06 |           235061 |                   nan |                   nan |                          nan |                  nan |                      22457 |                   2.73364e+06 |               nan |        1.43062e+12 |        1.44255e+12 | 2015-05-03T03:00:04.384000 | 2015-09-18T02:59:47.788000 |
| item_properties_combined | data/external/retailrocket/item_properties_part1.csv; data/external/retailrocket/item_properties_part2.csv |        851.865 |    2.02759e+07 |       nan           |           417053 |                  1104 |                   nan |                          nan |                  nan |                        nan |                 nan           |                 0 |        1.43123e+12 |        1.44211e+12 | 2015-05-10T03:00:00        | 2015-09-13T03:00:00        |
| category_tree            | data/external/retailrocket/category_tree.csv                                                               |          0.014 | 1669           |       nan           |              nan |                   nan |                  1669 |                          362 |                   25 |                        nan |                 nan           |               nan |      nan           |      nan           | nan                        | nan                        |

## 4.3 Retailrocket Event Distribution
| event       |   row_count |   distinct_visitors |   distinct_items |   rows_with_transaction_id |
|:------------|------------:|--------------------:|-----------------:|---------------------------:|
| view        |     2664312 |             1404179 |           234838 |                          0 |
| addtocart   |       69332 |               37722 |            23903 |                          0 |
| transaction |       22457 |               11719 |            12025 |                      22457 |

# 5. Pipeline Architecture
```text
Raw/API/External Data
  → Inspection
  → Validation
  → Staged Parquet
  → DuckDB Warehouse
  → Feature Tables
  → Model Training + MLflow
  → Model Artifacts + DVC
  → Inference Demo
  → Prefect Orchestration
```

# 6. Data Quality Validation

## 6.1 DummyJSON Validation Summary
| status   |   check_count |
|:---------|--------------:|
| PASS     |            46 |
| WARN     |             1 |

## 6.2 Retailrocket Validation Summary
| status   |   check_count |
|:---------|--------------:|
| PASS     |            36 |
Retailrocket validation checks include file existence, required columns, row counts, valid event types, transaction ID consistency, category metadata coverage, availability metadata coverage, and uniqueness of latest item metadata records.

# 7. DuckDB Warehouse
DuckDB is used as a local analytical warehouse. Staged Parquet files are loaded into structured tables and transformed into mart-level recommendation tables.
| schema_name   | relation_name                         | relation_type   |   row_count |   column_count | status   |
|:--------------|:--------------------------------------|:----------------|------------:|---------------:|:---------|
| staged        | retailrocket_events                   | BASE TABLE      |     2756101 |              8 | PASS     |
| staged        | retailrocket_category_tree            | BASE TABLE      |        1669 |              5 | PASS     |
| staged        | retailrocket_item_properties_selected | BASE TABLE      |     2291853 |              7 | PASS     |
| staged        | retailrocket_item_category_latest     | BASE TABLE      |      417053 |              6 | PASS     |
| staged        | retailrocket_item_availability_latest | BASE TABLE      |      417053 |              7 | PASS     |
| mart          | retailrocket_interactions             | VIEW            |     2756101 |              9 | PASS     |
| mart          | retailrocket_user_item_interactions   | BASE TABLE      |     2145179 |             14 | PASS     |
| mart          | retailrocket_user_features            | BASE TABLE      |     1407580 |             12 | PASS     |
| mart          | retailrocket_item_features            | BASE TABLE      |      235061 |             18 | PASS     |

# 8. Feature Engineering

## 8.1 DummyJSON Feature Summary
| schema_name   | table_name                     | relation_type   | output_file                                                 |   row_count |   column_count | output_exists   | status   |   missing_catalog_metadata_interactions |
|:--------------|:-------------------------------|:----------------|:------------------------------------------------------------|------------:|---------------:|:----------------|:---------|----------------------------------------:|
| features      | dummyjson_user_features        | BASE TABLE      | data/features/source=dummyjson/user_features.parquet        |         100 |             14 | True            | PASS     |                                     nan |
| features      | dummyjson_item_features        | BASE TABLE      | data/features/source=dummyjson/item_features.parquet        |         100 |             21 | True            | PASS     |                                     nan |
| features      | dummyjson_interaction_features | BASE TABLE      | data/features/source=dummyjson/interaction_features.parquet |         378 |             16 | True            | PASS     |                                     183 |

## 8.2 Retailrocket Feature Summary
| schema_name   | table_name                      | output_file                                                  |   row_count |   column_count |   file_size_mb | status   |
|:--------------|:--------------------------------|:-------------------------------------------------------------|------------:|---------------:|---------------:|:---------|
| features      | retailrocket_user_features      | data/features/source=retailrocket/user_features.parquet      |     1407580 |             16 |         47.328 | PASS     |
| features      | retailrocket_item_features      | data/features/source=retailrocket/item_features.parquet      |      235061 |             21 |          9.557 | PASS     |
| features      | retailrocket_user_item_features | data/features/source=retailrocket/user_item_features.parquet |     2145179 |             24 |         83.627 | PASS     |
Retailrocket features include user-level behavior, item-level popularity and metadata, and user-item interaction features such as event counts, implicit labels, and strongest event type.

# 9. Feature Registry
A custom feature registry was implemented for DummyJSON to demonstrate feature store concepts. It maps entities, feature views, source Parquet paths, and intended use cases such as training, batch inference, and candidate filtering.

# 10. Model Training

## 10.1 DummyJSON Popularity Baseline
| model_name             | model_type                 |   top_k |   total_interactions |   train_interactions |   test_interactions |   unique_users |   unique_items_in_interactions |   catalog_items |   recommended_item_count |   recommended_catalog_item_count |   catalog_item_coverage |   interaction_item_coverage |   evaluated_users |   hits |   hit_rate_at_10 |   precision_at_10 |   recall_at_10 |
|:-----------------------|:---------------------------|--------:|---------------------:|---------------------:|--------------------:|---------------:|-------------------------------:|----------------:|-------------------------:|---------------------------------:|------------------------:|----------------------------:|------------------:|-------:|-----------------:|------------------:|---------------:|
| popularity_recommender | global_popularity_baseline |      10 |                  378 |                  278 |                 100 |            100 |                            171 |             100 |                      150 |                               80 |                     0.8 |                    0.877193 |               100 |      2 |             0.02 |             0.002 |           0.02 |

## 10.2 Retailrocket Popularity Baseline
| model_name                          | model_type                       |   top_k |   test_window_days |   evaluation_sample_users |   min_timestamp_ms |   max_timestamp_ms |   cutoff_timestamp_ms | min_event_datetime               | max_event_datetime               |   train_event_count |   test_event_count |   train_distinct_items |   model_saved_item_count |   model_item_coverage |   test_target_users |   evaluated_users |   hits |   hit_rate_at_10 |   precision_at_10 |   recall_at_10 |   ndcg_at_10 |
|:------------------------------------|:---------------------------------|--------:|-------------------:|--------------------------:|-------------------:|-------------------:|----------------------:|:---------------------------------|:---------------------------------|--------------------:|-------------------:|-----------------------:|-------------------------:|----------------------:|--------------------:|------------------:|-------:|-----------------:|------------------:|---------------:|-------------:|
| retailrocket_popularity_recommender | event_weighted_global_popularity |      10 |                 14 |                     10000 |      1430622004384 |      1442545187788 |         1441335587788 | 2015-05-03 08:30:04.384000+05:30 | 2015-09-18 08:29:47.788000+05:30 |             2509138 |             246963 |                 225427 |                     5000 |             0.0221801 |                3648 |              3648 |     49 |         0.013432 |         0.0013432 |       0.013432 |   0.00804606 |

## 10.3 Retailrocket Content-Based Recommender
| model_name                             | model_type                       |   top_k |   test_window_days |   evaluation_sample_users |   min_timestamp_ms |   max_timestamp_ms |   cutoff_timestamp_ms |   candidate_item_count |   test_target_users |   user_profile_count |   evaluated_users |   hits |   hit_rate_at_10 |   precision_at_10 |   recall_at_10 |   ndcg_at_10 |
|:---------------------------------------|:---------------------------------|--------:|-------------------:|--------------------------:|-------------------:|-------------------:|----------------------:|-----------------------:|--------------------:|---------------------:|------------------:|-------:|-----------------:|------------------:|---------------:|-------------:|
| retailrocket_content_based_recommender | category_content_based_filtering |      10 |                 14 |                     10000 |      1430622004384 |      1442545187788 |         1441335587788 |                   5000 |                3648 |                  517 |              3648 |     39 |        0.0106908 |        0.00106908 |      0.0106908 |   0.00429383 |
The content-based recommender uses item category, parent category, availability metadata, conversion signals, and user category profiles to recommend items similar to a user's historical interests. This directly satisfies the assignment requirement for a content-based recommendation model.

## 10.4 Retailrocket Model Comparison
| model_family               | model_name                             | model_type                       |   top_k |   evaluated_users |   hits |   hit_rate_at_10 |   precision_at_10 |   recall_at_10 |   ndcg_at_10 |
|:---------------------------|:---------------------------------------|:---------------------------------|--------:|------------------:|-------:|-----------------:|------------------:|---------------:|-------------:|
| global_popularity_baseline | retailrocket_popularity_recommender    | event_weighted_global_popularity |      10 |              3648 |     49 |        0.013432  |        0.0013432  |      0.013432  |   0.00804606 |
| content_based_filtering    | retailrocket_content_based_recommender | category_content_based_filtering |      10 |              3648 |     39 |        0.0106908 |        0.00106908 |      0.0106908 |   0.00429383 |
The Retailrocket modeling stage includes two models. The first is an event-weighted global popularity baseline. The second is a category/content-based recommender that builds user profiles from historical category interactions and ranks candidate items using category similarity, metadata, conversion rates, and popularity prior. Event weights are: `view = 1`, `addtocart = 3`, and `transaction = 5`.

# 11. Model Evaluation
The Retailrocket popularity baseline evaluated 3648 users and achieved HitRate@10 = 0.013432, Precision@10 = 0.001343, Recall@10 = 0.013432, NDCG@10 = 0.008046.
The Retailrocket content-based recommender evaluated 3648 users and achieved HitRate@10 = 0.010691, Precision@10 = 0.001069, Recall@10 = 0.010691, NDCG@10 = 0.004294.
NDCG@10 is included because it measures ranking quality. A hit at rank 1 receives more credit than a hit at rank 10.

# 12. MLflow Experiment Tracking
MLflow was used to log model parameters, metrics, reports, and artifacts. Screenshots of the UI are stored in `reports/screenshots/`.

# 13. DVC Versioning
DVC tracks external data, staged data, feature data, and model artifacts. This keeps Git lightweight while preserving reproducibility for large files.

# 14. Inference Demo
| model_name                          | model_type                       |   requested_user_count |   top_k |   recommendation_rows |   distinct_recommended_items |   model_saved_item_count |   training_hit_rate_at_10 |   training_precision_at_10 |   training_recall_at_10 |   train_event_count |   test_event_count |   test_target_users |
|:------------------------------------|:---------------------------------|-----------------------:|--------:|----------------------:|-----------------------------:|-------------------------:|--------------------------:|---------------------------:|------------------------:|--------------------:|-------------------:|--------------------:|
| retailrocket_popularity_recommender | event_weighted_global_popularity |                      3 |       5 |                    15 |                            5 |                     5000 |                  0.013432 |                  0.0013432 |                0.013432 |             2509138 |             246963 |                3648 |
The Retailrocket inference script loads the trained model and generates recommendations for default active users, a single user, or custom comma-separated users. Previously interacted items are filtered.

# 15. Orchestration

## 15.1 DummyJSON Prefect Flow
|   step_order | step_name                    | status   |   duration_seconds |
|-------------:|:-----------------------------|:---------|-------------------:|
|            1 | ingest_dummyjson_api         | PASS     |              3.517 |
|            2 | inspect_dummyjson_raw        | PASS     |              0.459 |
|            3 | validate_dummyjson_raw       | PASS     |              0.374 |
|            4 | prepare_dummyjson_staged     | PASS     |              0.457 |
|            5 | validate_dummyjson_staged    | PASS     |              0.437 |
|            6 | load_dummyjson_to_duckdb     | PASS     |              0.729 |
|            7 | build_dummyjson_features     | PASS     |              0.46  |
|            8 | retrieve_dummyjson_features  | PASS     |              0.425 |
|            9 | train_popularity_recommender | PASS     |              2.759 |
|           10 | run_popularity_inference     | PASS     |              0.399 |

## 15.2 Retailrocket Prefect Flow
|   step_order | step_name                                 | status   |   duration_seconds |
|-------------:|:------------------------------------------|:---------|-------------------:|
|            1 | inspect_retailrocket_external             | PASS     |              4.668 |
|            2 | prepare_retailrocket_staged               | PASS     |              3.435 |
|            3 | validate_retailrocket_staged              | PASS     |              0.643 |
|            4 | load_retailrocket_to_duckdb               | PASS     |              5.156 |
|            5 | build_retailrocket_features               | PASS     |              4.57  |
|            6 | train_retailrocket_popularity_recommender | PASS     |             24.038 |
|            7 | run_retailrocket_inference                | PASS     |              0.585 |

# 16. Reproducibility
The project can be reproduced by activating the environment, checking DVC state, restoring artifacts if a DVC remote is configured, and running the Prefect orchestration scripts.

# 17. Team Work Division
| Team Member | Responsibility |
|---|---|
| Member 1 | Data ingestion and inspection |
| Member 2 | Data preparation, validation, and DVC tracking |
| Member 3 | DuckDB warehouse and feature engineering |
| Member 4 | MLflow, training, inference, orchestration, and final reporting |


# 18. Limitations and Future Work
- Current Retailrocket models are lightweight popularity and content-based baselines
- Add personalized collaborative filtering
- Add matrix factorization or item-item recommendations
- Add FastAPI serving endpoint
- Add Streamlit monitoring dashboard
- Add scheduled orchestration

# 19. Conclusion
The project successfully demonstrates a reproducible ML data management pipeline for recommendation systems using modern tools such as DuckDB, DVC, MLflow, and Prefect.