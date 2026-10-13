"""
train_models.py
---------------
Explores both datasets, trains Linear Regression on each, prints the
results and saves the figures used in the lab report.

Run:  python train_models.py
"""

import os

import matplotlib
matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from regression_core import DATASETS, train

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")


def banner(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def run(name):
    cfg = DATASETS[name]
    b = train(name)
    df, target = b["df"], cfg["target"]

    banner(f"DATASET: {cfg['title']}")
    print("Shape (rows, columns):", df.shape)

    print("\n--- First 5 rows ---")
    print(df.head())

    print("\n--- Missing values per column ---")
    print(df.isna().sum().to_string())

    print("\n--- Statistical summary ---")
    print(df.describe().round(3))

    print("\n--- Correlation of every feature with the target ---")
    print(df.corr()[target].drop(target).sort_values(ascending=False).round(3))

    banner(f"TRAINING LINEAR REGRESSION  ({cfg['title']})")
    print(f"Training samples : {len(b['X_train'])}")
    print(f"Testing samples  : {len(b['X_test'])}")

    model = b["model"]
    print(f"\nIntercept: {model.intercept_:.4f}")
    print("Coefficients:")
    for f, c in zip(b["features"], model.coef_):
        print(f"  {f:<14} {c:>12.4f}")

    m = b["metrics"]
    print("\nEvaluation on the test set:")
    print(f"  R2 score (train) : {m['train_r2']:.4f}")
    print(f"  R2 score (test)  : {m['test_r2']:.4f}")
    print(f"  MAE              : {m['mae']:.4f}")
    print(f"  MSE              : {m['mse']:.4f}")
    print(f"  RMSE             : {m['rmse']:.4f}")

    print("\nSample predictions (first 5 test rows):")
    sample = pd.DataFrame({
        "Actual": b["y_test"].values[:5],
        "Predicted": b["test_pred"][:5],
    })
    sample["Error"] = sample["Actual"] - sample["Predicted"]
    print(sample.round(2).to_string(index=False))

    # ------------------------------------------------------------------ plots
    # 1) target distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df[target], kde=True, ax=ax, color="#2a6f97")
    ax.set_title(f"Distribution of target: {target}")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/{name}_target_dist.png", dpi=150)
    plt.close(fig)

    # 2) correlation heatmap
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, ax=ax, annot_kws={"size": 7})
    ax.set_title("Correlation heatmap")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/{name}_heatmap.png", dpi=150)
    plt.close(fig)

    # 3) actual vs predicted
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(b["y_test"], b["test_pred"], alpha=0.7, color="#2a6f97",
               edgecolor="white")
    lo = min(b["y_test"].min(), b["test_pred"].min())
    hi = max(b["y_test"].max(), b["test_pred"].max())
    ax.plot([lo, hi], [lo, hi], "r--", label="Perfect prediction")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Actual vs Predicted (R2 = {m['test_r2']:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/{name}_actual_vs_pred.png", dpi=150)
    plt.close(fig)

    # 4) residuals
    residuals = b["y_test"].values - b["test_pred"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(b["test_pred"], residuals, alpha=0.7, color="#bc4749",
               edgecolor="white")
    ax.axhline(0, color="black", linestyle="--")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title("Residual plot")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/{name}_residuals.png", dpi=150)
    plt.close(fig)

    # 5) coefficients
    coefs = pd.Series(model.coef_, index=b["features"]).sort_values()
    fig, ax = plt.subplots(figsize=(6, 4))
    coefs.plot.barh(ax=ax, color="#386641")
    ax.set_title("Linear regression coefficients")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/{name}_coefficients.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved 5 figures to ./{FIG_DIR}/ with prefix '{name}_'")


if __name__ == "__main__":
    for key in DATASETS:
        run(key)
    print("\nDone.")
