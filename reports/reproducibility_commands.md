# Reproducibility Commands

## Activate environment

```bash
conda activate recomart
```

## Check repository and DVC state

```bash
git status
dvc status
```

## Restore DVC-tracked data and models if needed

```bash
dvc pull
```

## Run current Retailrocket pipeline

```bash
python -m orchestration.retailrocket_pipeline
```

## Run Retailrocket training only

```bash
python -m src.training.train_retailrocket_popularity_recommender
python -m src.training.train_retailrocket_content_recommender
```

## Run Retailrocket inference

```bash
python -m src.serving.recommend_retailrocket
python -m src.serving.recommend_retailrocket --user-id 1327109 --top-k 10
python -m src.serving.recommend_retailrocket --demo-users 1327109,925350,839657 --top-k 5
```

## Check DuckDB warehouse

```bash
duckdb data/warehouse/recomart.duckdb "SELECT event_type, COUNT(*) FROM mart.retailrocket_interactions GROUP BY event_type;"
```

## Regenerate reports

```bash
python -m src.reporting.generate_assignment_evidence
python -m src.reporting.generate_model_comparison
python -m src.reporting.generate_final_report
python -m src.reporting.generate_assignment_pdf
```