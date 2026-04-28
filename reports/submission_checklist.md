# RecoMart Submission Checklist

Use this checklist before creating the final ZIP and recording the walkthrough.

## Required Submission Items

- [x] Source code present under `src/`, `orchestration/`, `sql/`, and `configs/`
- [x] Consolidated PDF report present: `reports/recomart_assignment_report.pdf`
- [x] Markdown final report present: `reports/final_project_report.md`
- [x] Reproducibility commands present: `reports/reproducibility_commands.md`
- [x] Video walkthrough script present: `reports/video_walkthrough_script.md`
- [x] DVC metadata present for external, raw, staged, curated, feature, and model artifacts
- [x] MLflow evidence screenshots present under `reports/screenshots/`
- [x] Prefect orchestration evidence present: `reports/orchestration_retailrocket_pipeline_summary.csv`
- [x] Model comparison report present: `reports/retailrocket_model_comparison.csv`
- [x] Feature store registry present: `configs/feature_store/retailrocket_feature_registry.json`
- [x] Feature store retrieval evidence present: `reports/feature_store_retrieval_demo.csv`
- [x] Data quality reports present under `reports/data_quality/`

## Final Local Checks

- [ ] Git status clean after final approved commits
- [ ] DVC status clean
- [ ] Final PDF reviewed visually
- [ ] Video walkthrough recorded and checked for audio/readability
- [ ] Final ZIP pending explicit approval

## Do Not Include Directly in Git

- [x] Raw generated data payloads are DVC-tracked, not Git-tracked
- [x] Staged/generated Parquet payloads are DVC-tracked, not Git-tracked
- [x] Curated/generated Parquet payloads are DVC-tracked, not Git-tracked
- [x] Feature Parquet payloads are DVC-tracked, not Git-tracked
- [x] Model `.pkl` payloads are DVC-tracked, not Git-tracked
