"""Automated tests for the regime-aware monsoon rainfall post-processing pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.bias_correction import (
    apply_regime_aware_correction,
    build_residual_model,
    compute_mean_training_residual,
    uniform_correct,
)
from src.data_preprocessing import (
    FEATURES,
    REGIME,
    TARGET,
    VALID_REGIMES,
    make_demo_data,
    split_data,
    validate_data,
)
from src.evaluation import (
    classification_metrics,
    compute_skill_scores,
    evaluate_by_regime,
    regression_metrics,
    skill_score,
)
from src.regime_classifier import (
    build_regime_classifier,
    generate_oof_regime_predictions,
)
from train import run_training_pipeline


def test_make_demo_data_schema_and_bounds():
    df = make_demo_data(n_samples=200, random_state=7)
    assert len(df) == 200
    for col in FEATURES + [TARGET, REGIME]:
        assert col in df.columns
    assert (df["rainfall_forecast"] >= 0).all()
    assert (df[TARGET] >= 0).all()
    assert set(df[REGIME].unique()).issubset(set(VALID_REGIMES))


def test_validate_data_cleans_invalid_values_and_preserves_predictor_nans():
    df = make_demo_data(n_samples=80, random_state=11)
    # Inject invalid predictor values that should become NaN for training-only imputation
    df.loc[0, "humidity"] = 140.0
    df.loc[1, "wind_speed"] = -3.0
    # Inject case variation in regime
    df.loc[2, REGIME] = " active "
    # Inject invalid target and invalid regime rows that should be dropped
    df.loc[3, TARGET] = -5.0
    df.loc[4, REGIME] = "UnknownPhase"

    cleaned = validate_data(df)
    assert len(cleaned) == 78
    assert pd.isna(cleaned.loc[0, "humidity"])
    assert pd.isna(cleaned.loc[1, "wind_speed"])
    assert cleaned.loc[2, REGIME] == "Active"
    assert cleaned.attrs["cleaning_report"]["dropped_rows"] == 2
    assert cleaned.attrs["cleaning_report"]["invalid_predictor_values_coerced_to_nan"] == 2


def test_validate_data_rejects_missing_columns():
    df = make_demo_data(n_samples=60).drop(columns=["humidity"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_data(df)


def test_split_data_chronological_when_dates_present():
    df = make_demo_data(n_samples=120, random_state=1, include_dates=True)
    # Shuffle rows to ensure split_data sorts chronologically before splitting
    shuffled = df.sample(frac=1.0, random_state=99).reset_index(drop=True)
    cleaned = validate_data(shuffled)
    X_train, X_test, y_train, y_test, r_train, r_test, strategy = split_data(
        cleaned, test_size=0.25, split_strategy="auto"
    )
    assert strategy == "chronological"
    assert len(X_train) == 90
    assert len(X_test) == 30
    assert cleaned.loc[X_train.index.max(), "date"] < cleaned.loc[X_test.index.min(), "date"]


def test_regime_classifier_handles_missing_features_and_oof():
    df = validate_data(make_demo_data(n_samples=180, random_state=3))
    X_train, X_test, _, _, r_train, r_test, _ = split_data(df, test_size=0.2, random_state=3)

    X_train_missing = X_train.copy()
    X_train_missing.iloc[0, 1] = np.nan
    X_test_missing = X_test.copy()
    X_test_missing.iloc[0, 2] = np.nan

    oof = generate_oof_regime_predictions(X_train_missing, r_train, random_state=3)
    assert len(oof) == len(X_train_missing)
    assert set(oof).issubset(set(VALID_REGIMES))

    clf = build_regime_classifier(random_state=3)
    clf.fit(X_train_missing, r_train)
    preds = clf.predict(X_test_missing)
    proba = clf.predict_proba(X_test_missing)

    assert len(preds) == len(X_test_missing)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_bias_correction_clamps_non_negative():
    # Uniform correction should never output negative rainfall
    corrected = uniform_correct([2.0, 10.0], mean_training_residual=-5.0)
    assert np.allclose(corrected, [0.0, 5.0])

    # Regime-aware correction should handle NaNs and never output negative rainfall
    df = validate_data(make_demo_data(n_samples=150, random_state=5))
    X_train, X_test, y_train, _, r_train, _, _ = split_data(df, test_size=0.2, random_state=5)

    train_res = y_train.to_numpy() - X_train["rainfall_forecast"].to_numpy()
    model = build_residual_model(random_state=5)
    train_frame = X_train.copy()
    train_frame["predicted_regime"] = r_train.to_numpy()
    model.fit(train_frame, train_res)

    X_test_mod = X_test.copy()
    X_test_mod.iloc[0, 0] = 0.0  # zero raw forecast
    X_test_mod.iloc[1, 3] = np.nan
    preds = apply_regime_aware_correction(model, X_test_mod, ["Break"] * len(X_test_mod))
    assert (preds >= 0.0).all()
    assert not np.isnan(preds).any()


def test_evaluation_metrics_skill_scores_and_regime_breakdown():
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    raw = np.array([14.0, 24.0, 22.0, 32.0])
    uniform = np.array([12.0, 22.0, 24.0, 34.0])
    regime_aware = np.array([10.0, 20.0, 29.0, 41.0])
    regimes = np.array(["Break", "Break", "Active", "Active"])

    raw_m = regression_metrics(y_true, raw)
    reg_m = regression_metrics(y_true, regime_aware)
    assert reg_m["MAE"] < raw_m["MAE"]
    assert skill_score(2.0, 4.0) == pytest.approx(0.5)
    assert skill_score(5.0, 4.0) == pytest.approx(-0.25)
    assert skill_score(0.0, 0.0) == 0.0

    skills = compute_skill_scores(raw_m, reg_m)
    assert skills["MAE_skill_score"] > 0.0
    assert skills["RMSE_skill_score"] > 0.0

    cls_m = classification_metrics(["Active", "Break", "Other"], ["Active", "Break", "Active"])
    assert cls_m["accuracy"] == pytest.approx(2 / 3)
    assert len(cls_m["confusion_matrix"]["matrix"]) == 3

    by_regime = evaluate_by_regime(
        y_true,
        regimes,
        {
            "raw_forecast": raw,
            "uniform_bias_correction": uniform,
            "regime_aware_correction": regime_aware,
        },
    )
    assert set(by_regime.keys()) == {"Active", "Break"}
    assert by_regime["Active"]["support"] == 2
    assert by_regime["Break"]["methods"]["regime_aware_correction"]["MAE"] == pytest.approx(0.0)


def test_end_to_end_training_pipeline(tmp_path: Path):
    data_file = tmp_path / "synthetic.csv"
    make_demo_data(n_samples=300, random_state=42).to_csv(data_file, index=False)

    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    metrics = run_training_pipeline(
        data_path=data_file,
        test_size=0.2,
        random_state=42,
        models_dir=models_dir,
        reports_dir=reports_dir,
    )

    assert (models_dir / "monsoon_models.joblib").exists()
    assert (reports_dir / "metrics.json").exists()
    assert (reports_dir / "test_predictions.csv").exists()
    for plot_name in (
        "mae_comparison.png",
        "rmse_comparison.png",
        "regime_performance.png",
        "confusion_matrix.png",
        "actual_vs_predicted.png",
    ):
        plot_path = reports_dir / plot_name
        assert plot_path.exists()
        assert plot_path.stat().st_size > 0

    loaded_bundle = joblib.load(models_dir / "monsoon_models.joblib")
    assert set(["classifier", "residual_model", "uniform_residual"]).issubset(loaded_bundle.keys())

    loaded_metrics = json.loads((reports_dir / "metrics.json").read_text(encoding="utf-8"))
    assert loaded_metrics["dataset"]["rows_after_cleaning"] == 300
    assert "regime_specific_performance" in loaded_metrics
    assert "skill_scores_vs_raw" in loaded_metrics
    assert metrics["forecast_comparison"]["regime_aware_correction"]["MAE"] >= 0.0
