# RecoMart: Retailrocket Recommendation Data Pipeline

RecoMart is a reproducible ML data pipeline for Retailrocket recommendation data. It covers data inspection, validation, staged storage, DuckDB warehousing, feature engineering, model training, MLflow tracking, DVC versioning, inference, and Prefect orchestration.

# 1. Project Objective
The objective is to convert Retailrocket interaction and catalog metadata into validated features and trained recommender models for e-commerce recommendation use cases.

# 2. Tools Used
| Tool | Usage |
|---|---|
| Python | Main implementation language |
| pandas | Data inspection, summaries, reports |
| DuckDB | Local analytical warehouse and SQL transformations |
| Parquet | Staged and feature storage format |
| DVC | Versioning large datasets and model artifacts |
| Git | Versioning source code and lightweight reports |
| MLflow | Experiment tracking and metric logging |
| Prefect | Pipeline orchestration |


# 3. Repository Structure
```text
configs/          Feature registry configuration
data/external/    Retailrocket source archive tracked by DVC
data/raw/         Raw ingestion snapshots; populated in the ingestion refactor
data/staged/      Cleaned Retailrocket Parquet datasets tracked by DVC
data/curated/     Curated analytical datasets; populated in the preparation refactor
data/features/    Retailrocket ML feature datasets tracked by DVC
data/warehouse/   Generated DuckDB local warehouse
models/           Retailrocket model artifacts tracked by DVC
reports/          CSV summaries, plots, final report, screenshots
sql/              Retailrocket DuckDB SQL scripts
src/              Python modules
orchestration/    Prefect flows
```

# 4. Retailrocket Dataset Summary
| dataset_name             | file_path                                                                                                  |   file_size_mb |      row_count |   distinct_visitors |   distinct_items |   distinct_properties |   distinct_categories |   distinct_parent_categories |   root_category_rows |   rows_with_transaction_id |   rows_without_transaction_id |   null_value_rows |   min_timestamp_ms |   max_timestamp_ms | min_timestamp_datetime     | max_timestamp_datetime     |
|:-------------------------|:-----------------------------------------------------------------------------------------------------------|---------------:|---------------:|--------------------:|-----------------:|----------------------:|----------------------:|-----------------------------:|---------------------:|---------------------------:|------------------------------:|------------------:|-------------------:|-------------------:|:---------------------------|:---------------------------|
| events                   | data/external/retailrocket/events.csv                                                                      |         89.872 |    2.7561e+06  |         1.40758e+06 |           235061 |                   nan |                   nan |                          nan |                  nan |                      22457 |                   2.73364e+06 |               nan |        1.43062e+12 |        1.44255e+12 | 2015-05-03T03:00:04.384000 | 2015-09-18T02:59:47.788000 |
| item_properties_combined | data/external/retailrocket/item_properties_part1.csv; data/external/retailrocket/item_properties_part2.csv |        851.865 |    2.02759e+07 |       nan           |           417053 |                  1104 |                   nan |                          nan |                  nan |                        nan |                 nan           |                 0 |        1.43123e+12 |        1.44211e+12 | 2015-05-10T03:00:00        | 2015-09-13T03:00:00        |
| category_tree            | data/external/retailrocket/category_tree.csv                                                               |          0.014 | 1669           |       nan           |              nan |                   nan |                  1669 |                          362 |                   25 |                        nan |                 nan           |               nan |      nan           |      nan           | nan                        | nan                        |

## 4.1 Retailrocket Event Distribution
| event       |   row_count |   distinct_visitors |   distinct_items |   rows_with_transaction_id |
|:------------|------------:|--------------------:|-----------------:|---------------------------:|
| view        |     2664312 |             1404179 |           234838 |                          0 |
| addtocart   |       69332 |               37722 |            23903 |                          0 |
| transaction |       22457 |               11719 |            12025 |                      22457 |

# 5. Pipeline Architecture
```text
External Retailrocket Source Archive
  -> Raw Ingestion Layer
  -> Validation
  -> Staged Parquet
  -> Curated Analytical Data
  -> DuckDB Warehouse
  -> Feature Tables
  -> Model Training + MLflow
  -> Model Artifacts + DVC
  -> Inference Demo
  -> Prefect Orchestration
```

# 6. Data Quality Validation
| status   |   check_count |
|:---------|--------------:|
| PASS     |            36 |
Retailrocket validation checks include file existence, required columns, row counts, valid event types, transaction ID consistency, category metadata coverage, availability metadata coverage, and uniqueness of latest item metadata records.

# 7. DuckDB Warehouse
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
| schema_name   | table_name                      | output_file                                                  |   row_count |   column_count |   file_size_mb | status   |
|:--------------|:--------------------------------|:-------------------------------------------------------------|------------:|---------------:|---------------:|:---------|
| features      | retailrocket_user_features      | data/features/source=retailrocket/user_features.parquet      |     1407580 |             16 |         47.328 | PASS     |
| features      | retailrocket_item_features      | data/features/source=retailrocket/item_features.parquet      |      235061 |             21 |          9.557 | PASS     |
| features      | retailrocket_user_item_features | data/features/source=retailrocket/user_item_features.parquet |     2145179 |             24 |         83.627 | PASS     |
Retailrocket features include user-level behavior, item-level popularity and metadata, and user-item interaction features such as event counts, implicit labels, and strongest event type.

# 9. Feature Registry
A Retailrocket feature registry and retrieval demo are part of the planned feature-store refactor. The current feature tables are already generated under `data/features/source=retailrocket/`.

# 10. Model Training

## 10.1 Retailrocket Popularity Baseline
| model_name                          | model_type                       |   top_k |   test_window_days |   evaluation_sample_users |   min_timestamp_ms |   max_timestamp_ms |   cutoff_timestamp_ms | min_event_datetime               | max_event_datetime               |   train_event_count |   test_event_count |   train_distinct_items |   model_saved_item_count |   model_item_coverage |   test_target_users |   evaluated_users |   hits |   hit_rate_at_10 |   precision_at_10 |   recall_at_10 |   ndcg_at_10 |
|:------------------------------------|:---------------------------------|--------:|-------------------:|--------------------------:|-------------------:|-------------------:|----------------------:|:---------------------------------|:---------------------------------|--------------------:|-------------------:|-----------------------:|-------------------------:|----------------------:|--------------------:|------------------:|-------:|-----------------:|------------------:|---------------:|-------------:|
| retailrocket_popularity_recommender | event_weighted_global_popularity |      10 |                 14 |                     10000 |      1430622004384 |      1442545187788 |         1441335587788 | 2015-05-03 08:30:04.384000+05:30 | 2015-09-18 08:29:47.788000+05:30 |             2509138 |             246963 |                 225427 |                     5000 |             0.0221801 |                3648 |              3648 |     49 |         0.013432 |         0.0013432 |       0.013432 |   0.00804606 |

## 10.2 Retailrocket Content-Based Recommender
| model_name                             | model_type                       |   top_k |   test_window_days |   evaluation_sample_users |   min_timestamp_ms |   max_timestamp_ms |   cutoff_timestamp_ms |   candidate_item_count |   test_target_users |   user_profile_count |   evaluated_users |   hits |   hit_rate_at_10 |   precision_at_10 |   recall_at_10 |   ndcg_at_10 |
|:---------------------------------------|:---------------------------------|--------:|-------------------:|--------------------------:|-------------------:|-------------------:|----------------------:|-----------------------:|--------------------:|---------------------:|------------------:|-------:|-----------------:|------------------:|---------------:|-------------:|
| retailrocket_content_based_recommender | category_content_based_filtering |      10 |                 14 |                     10000 |      1430622004384 |      1442545187788 |         1441335587788 |                   5000 |                3648 |                  517 |              3648 |     39 |        0.0106908 |        0.00106908 |      0.0106908 |   0.00429383 |
The content-based recommender uses item category, parent category, availability metadata, conversion signals, and user category profiles to recommend items similar to a user's historical interests.

## 10.3 Retailrocket Model Comparison
| model_family               | model_name                             | model_type                       |   top_k |   evaluated_users |   hits |   hit_rate_at_10 |   precision_at_10 |   recall_at_10 |   ndcg_at_10 |
|:---------------------------|:---------------------------------------|:---------------------------------|--------:|------------------:|-------:|-----------------:|------------------:|---------------:|-------------:|
| global_popularity_baseline | retailrocket_popularity_recommender    | event_weighted_global_popularity |      10 |              3648 |     49 |        0.013432  |        0.0013432  |      0.013432  |   0.00804606 |
| content_based_filtering    | retailrocket_content_based_recommender | category_content_based_filtering |      10 |              3648 |     39 |        0.0106908 |        0.00106908 |      0.0106908 |   0.00429383 |

# 11. Model Evaluation
The Retailrocket popularity baseline evaluated 3648 users and achieved HitRate@10 = 0.013432, Precision@10 = 0.001343, Recall@10 = 0.013432, NDCG@10 = 0.008046.
The Retailrocket content-based recommender evaluated 3648 users and achieved HitRate@10 = 0.010691, Precision@10 = 0.001069, Recall@10 = 0.010691, NDCG@10 = 0.004294.

# 12. MLflow Experiment Tracking
MLflow logs Retailrocket model parameters, metrics, reports, and artifacts. Screenshots are stored in `reports/screenshots/`.

# 13. DVC Versioning
DVC tracks Retailrocket external data, staged data, feature data, and model artifacts. This keeps Git lightweight while preserving reproducibility for large files.

# 14. Inference Demo
| model_name                          | model_type                       |   requested_user_count |   top_k |   recommendation_rows |   distinct_recommended_items |   model_saved_item_count |   training_hit_rate_at_10 |   training_precision_at_10 |   training_recall_at_10 |   train_event_count |   test_event_count |   test_target_users |
|:------------------------------------|:---------------------------------|-----------------------:|--------:|----------------------:|-----------------------------:|-------------------------:|--------------------------:|---------------------------:|------------------------:|--------------------:|-------------------:|--------------------:|
| retailrocket_popularity_recommender | event_weighted_global_popularity |                      3 |       5 |                    15 |                            5 |                     5000 |                  0.013432 |                  0.0013432 |                0.013432 |             2509138 |             246963 |                3648 |

# 15. Orchestration
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
The project can be reproduced by activating the `recomart` environment, checking Git/DVC state, restoring artifacts if a DVC remote is configured, and running the Retailrocket pipeline commands.

# 17. Limitations and Future Work
- Add Retailrocket raw batch ingestion and REST catalog-delta ingestion
- Add curated analytical datasets
- Add a Retailrocket feature registry and retrieval demo
- Add scheduled REST ingestion with Prefect deployment documentation
- Regenerate the final assignment PDF after the refactor is complete

# 18. Conclusion
The current Retailrocket pipeline demonstrates a reproducible ML data management workflow and is being refactored into the final Retailrocket batch plus REST API assignment architecture.