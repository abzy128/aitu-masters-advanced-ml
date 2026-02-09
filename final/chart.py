"""
Chart generation from model training results.

Reads metrics JSON files from models/lstm/results/ and models/transformer/results/,
then produces comparison charts saved to results/ directory.
"""

import os
import json

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

LSTM_METRICS_PATH = os.path.join(
    PROJECT_ROOT, "models", "lstm", "results", "metrics.json"
)
TRANSFORMER_METRICS_PATH = os.path.join(
    PROJECT_ROOT, "models", "transformer", "results", "metrics.json"
)


def load_results(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def plot_training_loss(lstm: dict, transformer: dict, output_dir: str):
    """Plot training loss curves for both models side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    epochs_lstm = range(1, len(lstm["training_history"]["train_loss"]) + 1)
    epochs_tf = range(1, len(transformer["training_history"]["train_loss"]) + 1)

    # LSTM
    axes[0].plot(
        epochs_lstm,
        lstm["training_history"]["train_loss"],
        label="Train",
        linewidth=1.5,
    )
    axes[0].plot(
        epochs_lstm, lstm["training_history"]["test_loss"], label="Test", linewidth=1.5
    )
    axes[0].set_title("LSTM - Training & Test Loss", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Transformer
    axes[1].plot(
        epochs_tf,
        transformer["training_history"]["train_loss"],
        label="Train",
        linewidth=1.5,
    )
    axes[1].plot(
        epochs_tf,
        transformer["training_history"]["test_loss"],
        label="Test",
        linewidth=1.5,
    )
    axes[1].set_title(
        "Transformer - Training & Test Loss", fontsize=13, fontweight="bold"
    )
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Training Loss Curves", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "training_loss.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_loss_overlay(lstm: dict, transformer: dict, output_dir: str):
    """Overlay both models' test loss on the same plot for direct comparison."""
    fig, ax = plt.subplots(figsize=(10, 5))

    epochs_lstm = range(1, len(lstm["training_history"]["test_loss"]) + 1)
    epochs_tf = range(1, len(transformer["training_history"]["test_loss"]) + 1)

    ax.plot(
        epochs_lstm,
        lstm["training_history"]["test_loss"],
        label="LSTM",
        linewidth=1.8,
        color="#1f77b4",
    )
    ax.plot(
        epochs_tf,
        transformer["training_history"]["test_loss"],
        label="Transformer",
        linewidth=1.8,
        color="#ff7f0e",
    )

    ax.set_title("Test Loss Comparison", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "test_loss_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_metrics_comparison(lstm: dict, transformer: dict, output_dir: str):
    """Bar chart comparing final evaluation metrics between models."""
    metrics_names = ["MAE", "MSE", "RMSE", "R2"]
    metrics_keys = ["mae", "mse", "rmse", "r2"]

    lstm_vals = [lstm["metrics"][k] for k in metrics_keys]
    tf_vals = [transformer["metrics"][k] for k in metrics_keys]

    x = np.arange(len(metrics_names))
    width = 0.32

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(
        x - width / 2,
        lstm_vals,
        width,
        label="LSTM",
        color="#1f77b4",
        edgecolor="white",
    )
    bars2 = ax.bar(
        x + width / 2,
        tf_vals,
        width,
        label="Transformer",
        color="#ff7f0e",
        edgecolor="white",
    )

    # Value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(
            f"{height:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(
            f"{height:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title("Model Metrics Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=12)
    ax.set_ylabel("Value")
    ax.legend(fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "metrics_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_error_metrics_only(lstm: dict, transformer: dict, output_dir: str):
    """Bar chart of error metrics only (MAE, RMSE) — excluding MSE and R2
    for a cleaner scale comparison."""
    metrics_names = ["MAE", "RMSE"]
    metrics_keys = ["mae", "rmse"]

    lstm_vals = [lstm["metrics"][k] for k in metrics_keys]
    tf_vals = [transformer["metrics"][k] for k in metrics_keys]

    x = np.arange(len(metrics_names))
    width = 0.30

    fig, ax = plt.subplots(figsize=(7, 5))
    bars1 = ax.bar(
        x - width / 2,
        lstm_vals,
        width,
        label="LSTM",
        color="#1f77b4",
        edgecolor="white",
    )
    bars2 = ax.bar(
        x + width / 2,
        tf_vals,
        width,
        label="Transformer",
        color="#ff7f0e",
        edgecolor="white",
    )

    for bar in bars1:
        height = bar.get_height()
        ax.annotate(
            f"{height:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(
            f"{height:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_title(
        "Error Metrics Comparison (lower is better)", fontsize=13, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=12)
    ax.set_ylabel("Value (MW)")
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "error_metrics.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_r2_comparison(lstm: dict, transformer: dict, output_dir: str):
    """Horizontal bar chart for R2 score comparison."""
    models = ["LSTM", "Transformer"]
    r2_scores = [lstm["metrics"]["r2"], transformer["metrics"]["r2"]]
    colors = ["#1f77b4", "#ff7f0e"]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    bars = ax.barh(models, r2_scores, color=colors, height=0.45, edgecolor="white")

    for bar, val in zip(bars, r2_scores):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_xlim(0, 1.0)
    ax.set_title(
        "R\u00b2 Score Comparison (higher is better)", fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("R\u00b2")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "r2_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_training_summary(lstm: dict, transformer: dict, output_dir: str):
    """Summary table chart with model configs and final metrics."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    rows = [
        [
            "Parameters",
            f"{lstm['total_parameters']:,}",
            f"{transformer['total_parameters']:,}",
        ],
        [
            "Training Time",
            f"{lstm['training_time_seconds']:.1f}s",
            f"{transformer['training_time_seconds']:.1f}s",
        ],
        [
            "Final Train Loss",
            f"{lstm['training_history']['train_loss'][-1]:.6f}",
            f"{transformer['training_history']['train_loss'][-1]:.6f}",
        ],
        [
            "Final Test Loss",
            f"{lstm['training_history']['test_loss'][-1]:.6f}",
            f"{transformer['training_history']['test_loss'][-1]:.6f}",
        ],
        [
            "MAE",
            f"{lstm['metrics']['mae']:.4f}",
            f"{transformer['metrics']['mae']:.4f}",
        ],
        [
            "MSE",
            f"{lstm['metrics']['mse']:.4f}",
            f"{transformer['metrics']['mse']:.4f}",
        ],
        [
            "RMSE",
            f"{lstm['metrics']['rmse']:.4f}",
            f"{transformer['metrics']['rmse']:.4f}",
        ],
        [
            "R\u00b2",
            f"{lstm['metrics']['r2']:.4f}",
            f"{transformer['metrics']['r2']:.4f}",
        ],
    ]

    col_labels = ["Metric", "LSTM", "Transformer"]
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.6)

    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Alternate row colors
    for i in range(1, len(rows) + 1):
        color = "#f0f0f0" if i % 2 == 0 else "#ffffff"
        for j in range(len(col_labels)):
            table[i, j].set_facecolor(color)

    ax.set_title("Model Comparison Summary", fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    path = os.path.join(output_dir, "summary_table.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading results...")
    lstm = load_results(LSTM_METRICS_PATH)
    transformer = load_results(TRANSFORMER_METRICS_PATH)

    print(f"LSTM metrics: {lstm['metrics']}")
    print(f"Transformer metrics: {transformer['metrics']}")
    print()

    print("Generating charts...")
    plot_training_loss(lstm, transformer, RESULTS_DIR)
    plot_loss_overlay(lstm, transformer, RESULTS_DIR)
    plot_metrics_comparison(lstm, transformer, RESULTS_DIR)
    plot_error_metrics_only(lstm, transformer, RESULTS_DIR)
    plot_r2_comparison(lstm, transformer, RESULTS_DIR)
    plot_training_summary(lstm, transformer, RESULTS_DIR)

    print(f"\nAll charts saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
