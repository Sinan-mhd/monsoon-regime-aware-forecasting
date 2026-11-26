"""Reproducible end-to-end training and evaluation pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src.bias_correction import (
    add_regime_feature,
    apply_regime_aware_correction,
    build_residual_model,
    compute_mean_training_residual,
    uniform_correct,
)
from src.data_preprocessing import (
    FEATURES,
    REGIME,
    VALID_REGIMES,
    load_data,
    resolve_data_path,
    split_data,
    validate_data,
)
from src.evaluation import (
    classification_metrics,
    compute_skill_scores,
    evaluate_by_regime,
    plot_actual_vs_predicted,
    plot_confusion_matrix,
    plot_metric_comparison,
    plot_regime_performance,
    regression_metrics,
)
from src.regime_classifier import (
    build_regime_classifier,
    generate_oof_regime_predictions,
)


def _build_honest_interpretation(
    forecast_comparison: dict,
    skill_scores: dict,
    is_synthetic: bool,
) -> str:
    raw_mae = forecast_comparison["raw_forecast"]["MAE"]
    uni_mae = forecast_comparison["uniform_bias_correction"]["MAE"]
    reg_mae = forecast_comparison["regime_aware_correction"]["MAE"]

    raw_rmse = forecast_comparison["raw_forecast"]["RMSE"]
    uni_rmse = forecast_comparison["uniform_bias_correction"]["RMSE"]
    reg_rmse = forecast_comparison["regime_aware_correction"]["RMSE"]

    reg_mae_skill = skill_scores["regime_aware_correction"]["MAE_skill_score"]
    reg_rmse_skill = skill_scores["regime_aware_correction"]["RMSE_skill_score"]

    if reg_mae < min(raw_mae, uni_mae) and reg_rmse < min(raw_rmse, uni_rmse):
        perf_summary = (
            f"Regime-aware correction reduced both MAE ({raw_mae:.2f} -> {reg_mae:.2f} mm/day, "
            f"skill={reg_mae_skill:+.3f}) and RMSE ({raw_rmse:.2f} -> {reg_rmse:.2f} mm/day, "
            f"skill={reg_rmse_skill:+.3f}) relative to raw and uniform baselines."
        )
    else:
        perf_summary = (
            f"Regime-aware correction did not strictly outperform both baselines on all metrics "
            f"(Raw MAE={raw_mae:.2f}, Uniform MAE={uni_mae:.2f}, Regime-aware MAE={reg_mae:.2f}; "
            f"Raw RMSE={raw_rmse:.2f}, Uniform RMSE={uni_rmse:.2f}, Regime-aware RMSE={reg_rmse:.2f})."
        )

    provenance_note = (
        "These numbers come from synthetic demonstration data and verify pipeline behavior only; "
        "actual historical forecast and observation data are required for scientific validation."
        if is_synthetic
        else "Evaluate across independent seasons and stations before operational deployment."
    )
    return f"{perf_summary} {provenance_note}"


def run_training_pipeline(
    data_path: str | Path | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    split_strategy: str = "auto",
    models_dir: str | Path = "models",
    reports_dir: str | Path = "reports",
) -> dict:
    """Execute training, evaluation, model persistence, and plot generation."""
    resolved_path, is_synthetic = resolve_data_path(data_path)
    if is_synthetic:
        print(f"No user dataset found; generated synthetic demonstration dataset at: {resolved_path}")
    else:
        print(f"Loading dataset from: {resolved_path}")

    raw_df = load_data(resolved_path)
    df = validate_data(raw_df)
    cleaning_report = df.attrs.get("cleaning_report", {})

    X_train, X_test, y_train, y_test, r_train, r_test, used_split = split_data(
        df,
        test_size=test_size,
        random_state=random_state,
        split_strategy=split_strategy,
    )

    # Stage 1: Monsoon regime classification (fit strictly on training data)
    train_regime_oof = generate_oof_regime_predictions(
        X_train, r_train, random_state=random_state
    )
    classifier = build_regime_classifier(random_state=random_state)
    classifier.fit(X_train, r_train)
    test_regime_pred = classifier.predict(X_test)
    cls_metrics = classification_metrics(r_test, test_regime_pred, labels=VALID_REGIMES)

    # Stage 2A: Uniform bias correction baseline (training mean residual only)
    mean_training_residual = compute_mean_training_residual(
        y_train, X_train["rainfall_forecast"]
    )
    test_raw = X_test["rainfall_forecast"].to_numpy(dtype=float)
    test_uniform = uniform_correct(test_raw, mean_training_residual)

    # Stage 2B: Regime-aware residual correction (uses out-of-fold regimes on train, predicted regimes on test)
    train_residual = y_train.to_numpy(dtype=float) - X_train["rainfall_forecast"].to_numpy(dtype=float)
    residual_train_X = add_regime_feature(X_train, train_regime_oof)
    residual_model = build_residual_model(random_state=random_state, numeric_features=FEATURES)
    residual_model.fit(residual_train_X, train_residual)
    test_regime_aware = apply_regime_aware_correction(
        residual_model, X_test, test_regime_pred
    )

    # Overall & per-regime evaluation (test regimes used only after prediction for subgroup metrics)
    forecast_comparison = {
        "raw_forecast": regression_metrics(y_test, test_raw),
        "uniform_bias_correction": regression_metrics(y_test, test_uniform),
        "regime_aware_correction": regression_metrics(y_test, test_regime_aware),
    }
    overall_skill = {
        "uniform_bias_correction": compute_skill_scores(
            forecast_comparison["raw_forecast"],
            forecast_comparison["uniform_bias_correction"],
        ),
        "regime_aware_correction": compute_skill_scores(
            forecast_comparison["raw_forecast"],
            forecast_comparison["regime_aware_correction"],
        ),
    }
    regime_performance = evaluate_by_regime(
        y_test,
        r_test,
        {
            "raw_forecast": test_raw,
            "uniform_bias_correction": test_uniform,
            "regime_aware_correction": test_regime_aware,
        },
        labels=VALID_REGIMES,
    )

    regime_counts = {str(k): int(v) for k, v in df[REGIME].value_counts().items()}
    metrics = {
        "project": {
            "title": "Regime-Aware Post-Processing of Monsoon Rainfall Forecasts",
            "problem_statement_id": "SIH26080",
            "organisation": "Ministry of Earth Sciences (MoES)",
        },
        "dataset": {
            "source_path": resolved_path.as_posix(),
            "is_synthetic_demo": is_synthetic,
            "dataset_note": (
                "Synthetic demonstration dataset generated for software verification only; "
                "not real meteorological observations or IMD/MoES forecasts."
                if is_synthetic
                else "User-supplied dataset; verify provenance, forecast vintage, and regime-label methodology."
            ),
            "rows_after_cleaning": int(len(df)),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "split_strategy": used_split,
            "random_state": random_state,
            "regime_distribution": regime_counts,
            "cleaning_report": cleaning_report,
        },
        "training_summary": {
            "uniform_mean_training_residual_mm_day": mean_training_residual,
            "features": FEATURES,
            "regime_classes": list(VALID_REGIMES),
        },
        "regime_classification": cls_metrics,
        "regime_classification_accuracy": cls_metrics["accuracy"],
        "forecast_comparison": forecast_comparison,
        "skill_scores_vs_raw": overall_skill,
        "skill_score_definition": (
            "Skill Score = 1 - (Corrected Model Error / Raw Forecast Error). "
            "Positive values indicate error reduction relative to raw forecasts."
        ),
        "regime_specific_performance": regime_performance,
        "interpretation": _build_honest_interpretation(
            forecast_comparison, overall_skill, is_synthetic
        ),
    }

    models_path = Path(models_dir)
    reports_path = Path(reports_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    reports_path.mkdir(parents=True, exist_ok=True)

    artifact_path = models_path / "monsoon_models.joblib"
    joblib.dump(
        {
            "classifier": classifier,
            "residual_model": residual_model,
            "features": FEATURES,
            "regime_classes": list(VALID_REGIMES),
            "uniform_residual": mean_training_residual,
            "is_synthetic_demo": is_synthetic,
            "random_state": random_state,
            "split_strategy": used_split,
        },
        artifact_path,
    )

    metrics_file = reports_path / "metrics.json"
    metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    predictions_df = X_test.copy().reset_index(drop=True)
    predictions_df["observed_rainfall"] = y_test.to_numpy(dtype=float)
    predictions_df["true_regime"] = r_test.to_numpy().astype(str)
    predictions_df["predicted_regime"] = test_regime_pred.astype(str)
    predictions_df["raw_forecast"] = test_raw
    predictions_df["uniform_corrected"] = test_uniform
    predictions_df["regime_aware_corrected"] = test_regime_aware
    predictions_df.to_csv(reports_path / "test_predictions.csv", index=False)

    plot_metric_comparison(
        forecast_comparison,
        metric_key="MAE",
        ylabel="MAE (mm/day)",
        title="Raw vs Uniform vs Regime-Aware MAE",
        output_path=reports_path / "mae_comparison.png",
    )
    plot_metric_comparison(
        forecast_comparison,
        metric_key="RMSE",
        ylabel="RMSE (mm/day)",
        title="Raw vs Uniform vs Regime-Aware RMSE",
        output_path=reports_path / "rmse_comparison.png",
    )
    plot_regime_performance(
        regime_performance,
        output_path=reports_path / "regime_performance.png",
    )
    plot_confusion_matrix(
        cls_metrics["confusion_matrix"],
        output_path=reports_path / "confusion_matrix.png",
    )
    plot_actual_vs_predicted(
        y_test,
        test_raw,
        test_uniform,
        test_regime_aware,
        output_path=reports_path / "actual_vs_predicted.png",
    )

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train regime-aware monsoon rainfall post-processing models."
    )
    parser.add_argument(
        "--data",
        default=None,
        help=(
            "Path to CSV dataset. If omitted, uses data/monsoon_data.csv when present "
            "or generates synthetic data/demo_monsoon.csv."
        ),
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--split",
        choices=["auto", "chronological", "stratified"],
        default="auto",
        help="Train/test split strategy ('auto' uses chronological when a date column exists).",
    )
    args = parser.parse_args()

    metrics = run_training_pipeline(
        data_path=args.data,
        test_size=args.test_size,
        random_state=args.random_state,
        split_strategy=args.split,
    )
    print(json.dumps(metrics, indent=2))
    print(
        "\nSaved artifacts: models/monsoon_models.joblib, reports/metrics.json, "
        "reports/test_predictions.csv, and 5 evaluation plots in reports/."
    )


if __name__ == "__main__":
    main()
