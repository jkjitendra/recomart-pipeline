from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


REPORTS_DIR = Path("reports")
PLOTS_DIR = REPORTS_DIR / "plots"

POPULARITY_SUMMARY_PATH = REPORTS_DIR / "retailrocket_model_training_summary.csv"
CONTENT_SUMMARY_PATH = REPORTS_DIR / "retailrocket_content_model_training_summary.csv"

MODEL_COMPARISON_PATH = REPORTS_DIR / "retailrocket_model_comparison.csv"
MODEL_COMPARISON_PLOT_PATH = PLOTS_DIR / "retailrocket_model_comparison_metrics.png"


def load_model_summary(path: Path, model_family: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing model summary file: {path}")

    df = pd.read_csv(path)
    df["model_family"] = model_family
    return df


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    popularity_df = load_model_summary(
        POPULARITY_SUMMARY_PATH,
        "global_popularity_baseline",
    )

    content_df = load_model_summary(
        CONTENT_SUMMARY_PATH,
        "content_based_filtering",
    )

    comparison_df = pd.concat(
        [popularity_df, content_df],
        ignore_index=True,
        sort=False,
    )

    selected_columns = [
        "model_family",
        "model_name",
        "model_type",
        "top_k",
        "evaluated_users",
        "hits",
        "hit_rate_at_10",
        "precision_at_10",
        "recall_at_10",
        "ndcg_at_10",
    ]

    selected_columns = [
        column for column in selected_columns
        if column in comparison_df.columns
    ]

    comparison_df = comparison_df[selected_columns]
    comparison_df.to_csv(MODEL_COMPARISON_PATH, index=False)

    metrics = [
        "hit_rate_at_10",
        "precision_at_10",
        "recall_at_10",
        "ndcg_at_10",
    ]

    plot_df = comparison_df.set_index("model_family")[metrics].T

    ax = plot_df.plot(kind="bar", figsize=(10, 6))
    ax.set_title("Retailrocket Model Comparison")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Model")
    plt.tight_layout()
    plt.savefig(MODEL_COMPARISON_PLOT_PATH, dpi=160)
    plt.close()

    print(f"Model comparison saved: {MODEL_COMPARISON_PATH}")
    print(f"Model comparison plot saved: {MODEL_COMPARISON_PLOT_PATH}")
    print()
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()