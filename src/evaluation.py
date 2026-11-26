"""Evaluation metrics, skill scores, per-regime breakdowns, and Matplotlib figures."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
)

from src.data_preprocessing import VALID_REGIMES


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Compute MAE and RMSE for rainfall forecasts."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    return {
        "MAE": float(mean_absolute_error(y_t, y_p)),
        "RMSE": float(np.sqrt(mean_squared_error(y_t, y_p))),
    }


def skill_score(corrected_error: float, raw_error: float) -> float:
    """Compute skill score: 1 - (Corrected Model Error / Raw Forecast Error).

    Positive values indicate improvement over raw forecasts.
    """
    if raw_error <= 0.0:
        return 0.0
    return float(1.0 - (corrected_error / raw_error))


def compute_skill_scores(
    raw_metrics: Mapping[str, float],
    corrected_metrics: Mapping[str, float],
) -> dict[str, float]:
    """Compute MAE and RMSE skill scores relative to raw forecasts."""
    return {
        "MAE_skill_score": skill_score(corrected_metrics["MAE"], raw_metrics["MAE"]),
        "RMSE_skill_score": skill_score(corrected_metrics["RMSE"], raw_metrics["RMSE"]),
    }


def classification_accuracy(y_true, y_pred) -> float:
    """Compute overall regime classification accuracy."""
    return float(accuracy_score(y_true, y_pred))


def classification_metrics(
    y_true,
    y_pred,
    labels: Sequence[str] = VALID_REGIMES,
) -> dict:
    """Compute accuracy, macro/weighted precision/recall/F1, per-class metrics, and confusion matrix."""
    y_t = np.asarray(y_true).astype(str)
    y_p = np.asarray(y_pred).astype(str)
    present_labels = [lbl for lbl in labels if lbl in set(y_t) | set(y_p)]
    if not present_labels:
        present_labels = sorted(set(y_t) | set(y_p))

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_t, y_p, labels=present_labels, average="macro", zero_division=0
    )
    prec_wt, rec_wt, f1_wt, _ = precision_recall_fscore_support(
        y_t, y_p, labels=present_labels, average="weighted", zero_division=0
    )
    prec_cls, rec_cls, f1_cls, sup_cls = precision_recall_fscore_support(
        y_t, y_p, labels=present_labels, average=None, zero_division=0
    )
    cm = confusion_matrix(y_t, y_p, labels=present_labels)

    per_class = {}
    for idx, lbl in enumerate(present_labels):
        per_class[lbl] = {
            "precision": float(prec_cls[idx]),
            "recall": float(rec_cls[idx]),
            "f1_score": float(f1_cls[idx]),
            "support": int(sup_cls[idx]),
        }

    return {
        "accuracy": float(accuracy_score(y_t, y_p)),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(prec_wt),
        "recall_weighted": float(rec_wt),
        "f1_weighted": float(f1_wt),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": present_labels,
            "matrix": cm.tolist(),
        },
    }


def evaluate_by_regime(
    y_true,
    regimes_true,
    predictions_by_method: Mapping[str, np.ndarray],
    labels: Sequence[str] = VALID_REGIMES,
) -> dict:
    """Evaluate raw, uniform, and regime-aware forecasts separately for each true test regime.

    Note:
        True test regimes are used strictly here for subgroup evaluation AFTER all
        model predictions have been generated without access to test regimes.
    """
    y_arr = np.asarray(y_true, dtype=float)
    r_arr = np.asarray(regimes_true).astype(str)
    results = {}

    ordered_regimes = [lbl for lbl in labels if lbl in set(r_arr)]
    for lbl in sorted(set(r_arr)):
        if lbl not in ordered_regimes:
            ordered_regimes.append(lbl)

    for regime_label in ordered_regimes:
        mask = r_arr == regime_label
        support = int(mask.sum())
        if support == 0:
            continue

        method_metrics = {}
        for method_name, preds in predictions_by_method.items():
            pred_arr = np.asarray(preds, dtype=float)
            method_metrics[method_name] = regression_metrics(y_arr[mask], pred_arr[mask])

        raw_m = method_metrics["raw_forecast"]
        skill = {}
        for method_name in ("uniform_bias_correction", "regime_aware_correction"):
            if method_name in method_metrics:
                skill[method_name] = compute_skill_scores(raw_m, method_metrics[method_name])

        results[regime_label] = {
            "support": support,
            "methods": method_metrics,
            "skill_scores_vs_raw": skill,
        }

    return results


def plot_metric_comparison(
    forecast_comparison: Mapping[str, Mapping[str, float]],
    metric_key: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    """Save bar chart comparing Raw, Uniform, and Regime-aware error metrics."""
    ordered = [
        ("raw_forecast", "Raw forecast", "#64748b"),
        ("uniform_bias_correction", "Uniform correction", "#f59e0b"),
        ("regime_aware_correction", "Regime-aware correction", "#0284c7"),
    ]
    labels = [display for _, display, _ in ordered]
    values = [forecast_comparison[key][metric_key] for key, _, _ in ordered]
    colors = [c for _, _, c in ordered]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bars = ax.bar(labels, values, color=colors, width=0.55, edgecolor="#1e293b", linewidth=0.7)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(values) * 1.18 if max(values) > 0 else 1.0)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + (max(values) * 0.025 if max(values) > 0 else 0.05),
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_regime_performance(regime_eval: Mapping, output_path: Path) -> None:
    """Save grouped bar chart of MAE and RMSE across monsoon regimes."""
    regimes = list(regime_eval.keys())
    methods = [
        ("raw_forecast", "Raw", "#64748b"),
        ("uniform_bias_correction", "Uniform", "#f59e0b"),
        ("regime_aware_correction", "Regime-aware", "#0284c7"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(regimes))
    width = 0.25

    for ax, metric_key, ylabel in (
        (axes[0], "MAE", "MAE (mm/day)"),
        (axes[1], "RMSE", "RMSE (mm/day)"),
    ):
        for offset_idx, (m_key, m_label, color) in enumerate(methods):
            vals = [regime_eval[r]["methods"][m_key][metric_key] for r in regimes]
            bars = ax.bar(
                x + (offset_idx - 1) * width,
                vals,
                width=width,
                label=m_label,
                color=color,
                edgecolor="#1e293b",
                linewidth=0.6,
            )
            for bar, v in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    v + 0.4,
                    f"{v:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                )

        ax.set_xticks(x)
        ax.set_xticklabels([f"{r}\n(n={regime_eval[r]['support']})" for r in regimes])
        ax.set_ylabel(ylabel, fontsize=10.5)
        ax.set_title(f"{metric_key} by Monsoon Regime", fontsize=11.5, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)
        ax.legend(frameon=False)

    fig.suptitle("Forecast Error Across Monsoon Phases", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm_dict: Mapping, output_path: Path) -> None:
    """Save annotated confusion matrix heatmap for monsoon regime classification."""
    labels = cm_dict["labels"]
    matrix = np.asarray(cm_dict["matrix"], dtype=int)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True Regime",
        xlabel="Predicted Regime",
        title="Monsoon Regime Classification Confusion Matrix",
    )

    thresh = matrix.max() / 2.0 if matrix.size > 0 else 0.0
    row_sums = matrix.sum(axis=1, keepdims=True)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            pct = (100.0 * matrix[i, j] / row_sums[i, 0]) if row_sums[i, 0] > 0 else 0.0
            ax.text(
                j,
                i,
                f"{matrix[i, j]}\n({pct:.1f}%)",
                ha="center",
                va="center",
                color="white" if matrix[i, j] > thresh else "#0f172a",
                fontsize=9.5,
                fontweight="bold",
            )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_actual_vs_predicted(
    y_true,
    raw_pred,
    uniform_pred,
    regime_pred,
    output_path: Path,
) -> None:
    """Save 3-panel actual vs predicted rainfall comparison scatter plot."""
    y_arr = np.asarray(y_true, dtype=float)
    panels = [
        ("Raw Forecast", np.asarray(raw_pred, dtype=float), "#64748b"),
        ("Uniform Bias Correction", np.asarray(uniform_pred, dtype=float), "#f59e0b"),
        ("Regime-Aware Correction", np.asarray(regime_pred, dtype=float), "#0284c7"),
    ]
    max_val = max(
        float(np.max(y_arr)),
        max(float(np.max(p)) for _, p, _ in panels),
        10.0,
    ) * 1.05

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.7), sharex=True, sharey=True)
    for ax, (title, preds, color) in zip(axes, panels):
        m = regression_metrics(y_arr, preds)
        ax.scatter(y_arr, preds, alpha=0.45, s=18, color=color, edgecolors="none")
        ax.plot([0, max_val], [0, max_val], linestyle="--", color="#dc2626", linewidth=1.3, label="1:1 Ideal")
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
        ax.set_xlabel("Observed Rainfall (mm/day)", fontsize=10)
        ax.set_title(f"{title}\nMAE={m['MAE']:.2f} | RMSE={m['RMSE']:.2f}", fontsize=10.5, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="upper left", frameon=False, fontsize=8.5)

    axes[0].set_ylabel("Predicted Rainfall (mm/day)", fontsize=10)
    fig.suptitle("Observed vs Predicted Rainfall Comparison (Test Set)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
