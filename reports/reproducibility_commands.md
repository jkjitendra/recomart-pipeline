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

If no DVC remote is configured, the DVC cache must already exist locally.

## Run DummyJSON pipeline

```bash
python -m orchestration.dummyjson_pipeline
```

## Run Retailrocket pipeline

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

## Open MLflow UI

Use the same MLflow tracking URI used by the training scripts.

```bash
TRACKING_URI=$(python -c 'import mlflow; print(mlflow.get_tracking_uri())')
mlflow ui --backend-store-uri "$TRACKING_URI" --port 5001
```

Open `http://127.0.0.1:5001`, then go to `model training → Experiments`.

## Check DuckDB warehouse

```bash
duckdb data/warehouse/recomart.duckdb "SELECT event_type, COUNT(*) FROM mart.retailrocket_interactions GROUP BY event_type;"
```

## Regenerate final report

```bash
python -m src.reporting.generate_assignment_evidence
python -m src.reporting.generate_model_comparison
python -m src.reporting.generate_final_report
python -m src.reporting.generate_assignment_pdf
```