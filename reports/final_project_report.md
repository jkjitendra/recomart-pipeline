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
| PASS     |            59 |
| WARN     |             3 |
Retailrocket validation checks include file existence, required columns, row counts, valid event types, transaction ID consistency, category metadata coverage, availability metadata coverage, and uniqueness of latest item metadata records.

# 7. DuckDB Warehouse
| schema_name   | relation_name                          | relation_type   |   row_count |   column_count | status   |
|:--------------|:---------------------------------------|:----------------|------------:|---------------:|:---------|
| staged        | retailrocket_events                    | BASE TABLE      |     2756101 |              8 | PASS     |
| staged        | retailrocket_category_tree             | BASE TABLE      |        1669 |              5 | PASS     |
| staged        | retailrocket_item_properties_batch     | BASE TABLE      |     1243692 |              8 | PASS     |
| staged        | retailrocket_item_properties_api_delta | BASE TABLE      |          40 |              9 | PASS     |
| staged        | retailrocket_item_properties_combined  | BASE TABLE      |     1243732 |              9 | PASS     |
| staged        | retailrocket_item_category_latest      | BASE TABLE      |      229342 |              6 | PASS     |
| staged        | retailrocket_item_availability_latest  | BASE TABLE      |      236884 |              7 | PASS     |
| curated       | retailrocket_interactions              | BASE TABLE      |     2756101 |             13 | PASS     |
| curated       | retailrocket_items                     | BASE TABLE      |      235061 |             16 | PASS     |
| curated       | retailrocket_user_item_interactions    | BASE TABLE      |     2145179 |             19 | PASS     |
| mart          | retailrocket_interactions              | VIEW            |     2756101 |             13 | PASS     |
| mart          | retailrocket_user_item_interactions    | BASE TABLE      |     2145179 |             19 | PASS     |
| mart          | retailrocket_user_features             | BASE TABLE      |     1407580 |             12 | PASS     |
| mart          | retailrocket_item_features             | BASE TABLE      |      235061 |             18 | PASS     |

# 8. Feature Engineering
| schema_name   | table_name                      | output_file                                                  |   row_count |   column_count |   file_size_mb | status   |
|:--------------|:--------------------------------|:-------------------------------------------------------------|------------:|---------------:|---------------:|:---------|
| features      | retailrocket_user_features      | data/features/source=retailrocket/user_features.parquet      |     1407580 |             16 |         47.35  | PASS     |
| features      | retailrocket_item_features      | data/features/source=retailrocket/item_features.parquet      |      235061 |             21 |          9.072 | PASS     |
| features      | retailrocket_user_item_features | data/features/source=retailrocket/user_item_features.parquet |     2145179 |             24 |         81.464 | PASS     |
Retailrocket features include user-level behavior, item-level popularity and metadata, and user-item interaction features such as event counts, implicit labels, and strongest event type.

# 9. Feature Registry
A Retailrocket feature registry documents user, item, and user-item feature views. The retrieval demo reads current feature Parquet files under `data/features/source=retailrocket/` and writes sample retrieval evidence.

# 10. Model Training

## 10.1 Retailrocket Popularity Baseline
| model_name                          | model_type                       |   top_k |   test_window_days |   evaluation_sample_users |   min_timestamp_ms |   max_timestamp_ms |   cutoff_timestamp_ms | min_event_datetime               | max_event_datetime               |   train_event_count |   test_event_count |   train_distinct_items |   model_saved_item_count |   model_item_coverage |   test_target_users |   evaluated_users |   hits |   hit_rate_at_10 |   precision_at_10 |   recall_at_10 |   ndcg_at_10 |
|:------------------------------------|:---------------------------------|--------:|-------------------:|--------------------------:|-------------------:|-------------------:|----------------------:|:---------------------------------|:---------------------------------|--------------------:|-------------------:|-----------------------:|-------------------------:|----------------------:|--------------------:|------------------:|-------:|-----------------:|------------------:|---------------:|-------------:|
| retailrocket_popularity_recommender | event_weighted_global_popularity |      10 |                 14 |                     10000 |      1430622004384 |      1442545187788 |         1441335587788 | 2015-05-03 08:30:04.384000+05:30 | 2015-09-18 08:29:47.788000+05:30 |             2509138 |             246963 |                 225427 |                     5000 |             0.0221801 |                3648 |              3648 |     49 |         0.013432 |         0.0013432 |       0.013432 |   0.00762866 |

## 10.2 Retailrocket Content-Based Recommender
| model_name                             | model_type                       |   top_k |   test_window_days |   evaluation_sample_users |   min_timestamp_ms |   max_timestamp_ms |   cutoff_timestamp_ms |   candidate_item_count |   test_target_users |   user_profile_count |   evaluated_users |   hits |   hit_rate_at_10 |   precision_at_10 |   recall_at_10 |   ndcg_at_10 |
|:---------------------------------------|:---------------------------------|--------:|-------------------:|--------------------------:|-------------------:|-------------------:|----------------------:|-----------------------:|--------------------:|---------------------:|------------------:|-------:|-----------------:|------------------:|---------------:|-------------:|
| retailrocket_content_based_recommender | category_content_based_filtering |      10 |                 14 |                     10000 |      1430622004384 |      1442545187788 |         1441335587788 |                   5000 |                3648 |                  517 |              3648 |     27 |       0.00740132 |       0.000740132 |     0.00740132 |   0.00281525 |
The content-based recommender uses item category, parent category, availability metadata, conversion signals, and user category profiles to recommend items similar to a user's historical interests.

## 10.3 Retailrocket Model Comparison
| model_family               | model_name                             | model_type                       |   top_k |   evaluated_users |   hits |   hit_rate_at_10 |   precision_at_10 |   recall_at_10 |   ndcg_at_10 | model_artifact_path                               | training_report_path                                    |
|:---------------------------|:---------------------------------------|:---------------------------------|--------:|------------------:|-------:|-----------------:|------------------:|---------------:|-------------:|:--------------------------------------------------|:--------------------------------------------------------|
| global_popularity_baseline | retailrocket_popularity_recommender    | event_weighted_global_popularity |      10 |              3648 |     49 |       0.013432   |       0.0013432   |     0.013432   |   0.00762866 | models/retailrocket/popularity_recommender.pkl    | reports/retailrocket_model_training_summary.csv         |
| content_based_filtering    | retailrocket_content_based_recommender | category_content_based_filtering |      10 |              3648 |     27 |       0.00740132 |       0.000740132 |     0.00740132 |   0.00281525 | models/retailrocket/content_based_recommender.pkl | reports/retailrocket_content_model_training_summary.csv |

# 11. Model Evaluation
The Retailrocket popularity baseline evaluated 3648 users and achieved HitRate@10 = 0.013432, Precision@10 = 0.001343, Recall@10 = 0.013432, NDCG@10 = 0.007629.
The Retailrocket content-based recommender evaluated 3648 users and achieved HitRate@10 = 0.007401, Precision@10 = 0.000740, Recall@10 = 0.007401, NDCG@10 = 0.002815.

# 12. MLflow Experiment Tracking
MLflow logs Retailrocket model parameters, metrics, reports, and artifacts. Screenshots are stored in `reports/screenshots/`.

# 13. DVC Versioning
DVC tracks Retailrocket external data, staged data, feature data, and model artifacts. This keeps Git lightweight while preserving reproducibility for large files.

# 14. Inference Demo
| model_name                             | model_type                       |   requested_user_count |   top_k |   recommendation_rows |   distinct_recommended_items |   training_hit_rate_at_10 |   training_precision_at_10 |   training_recall_at_10 |   test_target_users | model_key     | model_artifact_path                               |   candidate_or_saved_item_count |   training_ndcg_at_10 |   train_event_count |   test_event_count |
|:---------------------------------------|:---------------------------------|-----------------------:|--------:|----------------------:|-----------------------------:|--------------------------:|---------------------------:|------------------------:|--------------------:|:--------------|:--------------------------------------------------|--------------------------------:|----------------------:|--------------------:|-------------------:|
| retailrocket_popularity_recommender    | event_weighted_global_popularity |                      5 |       5 |                    25 |                           18 |                0.013432   |                0.0013432   |              0.013432   |                3648 | popularity    | models/retailrocket/popularity_recommender.pkl    |                            5000 |            0.00762866 |         2.50914e+06 |             246963 |
| retailrocket_content_based_recommender | category_content_based_filtering |                      5 |       5 |                    25 |                            7 |                0.00740132 |                0.000740132 |              0.00740132 |                3648 | content_based | models/retailrocket/content_based_recommender.pkl |                            5000 |            0.00281525 |       nan           |                nan |

# 15. Orchestration
|   step_order | step_name                                 | status   |   duration_seconds |
|-------------:|:------------------------------------------|:---------|-------------------:|
|            1 | inspect_retailrocket_external             | PASS     |              5.243 |
|            2 | ingest_retailrocket_batch                 | PASS     |              9.308 |
|            3 | ingest_retailrocket_catalog_api_delta     | PASS     |              0.939 |
|            4 | validate_retailrocket_raw                 | PASS     |              6.62  |
|            5 | prepare_retailrocket_staged_and_curated   | PASS     |              3.963 |
|            6 | validate_retailrocket_staged_and_curated  | PASS     |              0.818 |
|            7 | load_retailrocket_to_duckdb               | PASS     |              7.27  |
|            8 | build_retailrocket_features               | PASS     |              5.176 |
|            9 | retrieve_retailrocket_features            | PASS     |              5.535 |
|           10 | train_retailrocket_popularity_recommender | PASS     |             31.177 |
|           11 | train_retailrocket_content_recommender    | PASS     |             64.244 |
|           12 | run_retailrocket_popularity_inference     | PASS     |              0.979 |
|           13 | run_retailrocket_content_based_inference  | PASS     |              0.911 |
|           14 | generate_model_comparison                 | PASS     |              0.992 |
|           15 | generate_assignment_evidence              | PASS     |              1.31  |
|           16 | generate_final_report                     | PASS     |              0.586 |
|           17 | generate_assignment_pdf                   | PASS     |              1.907 |

# 16. Reproducibility
The project can be reproduced by activating the `recomart` environment, checking Git/DVC state, restoring artifacts if a DVC remote is configured, and running the Retailrocket pipeline commands.

# 17. Limitations and Future Work
- The current models are lightweight recommenders suitable for the assignment scope
- Future work can add collaborative filtering or sequence-aware recommendation models
- A production deployment would use a managed API endpoint and persistent Prefect work pool
- Phase 7 refreshes final narrative wording and PDF presentation for submission polish

# 18. Conclusion
The Retailrocket pipeline demonstrates a reproducible ML data management workflow with batch ingestion, REST catalog-delta ingestion, validation, preparation, warehousing, feature engineering, model training, MLflow tracking, inference, DVC versioning, and Prefect orchestration.