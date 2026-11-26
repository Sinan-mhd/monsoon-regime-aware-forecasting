"""Uniform and regime-aware rainfall forecast bias correction models."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data_preprocessing import FEATURES, VALID_REGIMES


def compute_mean_training_residual(y_train, forecast_train) -> float:
    """Compute mean residual (observed - raw_forecast) strictly on training samples."""
    y_arr = np.asarray(y_train, dtype=float)
    f_arr = np.asarray(forecast_train, dtype=float)
    return float(np.mean(y_arr - f_arr))


def uniform_correct(forecast, mean_training_residual: float) -> np.ndarray:
    """Apply uniform additive bias correction and clamp outputs to non-negative values."""
    corrected = np.asarray(forecast, dtype=float) + float(mean_training_residual)
    return np.maximum(0.0, corrected)


def add_regime_feature(X: pd.DataFrame, predicted_regimes) -> pd.DataFrame:
    """Attach predicted monsoon regime labels to predictor dataframe."""
    result = X.copy()
    result["predicted_regime"] = np.asarray(predicted_regimes).astype(str)
    return result


def build_residual_model(
    random_state: int = 42,
    numeric_features: Sequence[str] = FEATURES,
) -> Pipeline:
    """Build a leak-free scikit-learn pipeline predicting forecast residuals."""
    numeric_cols = list(numeric_features)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            (
                "regime",
                OneHotEncoder(
                    categories=[list(VALID_REGIMES)],
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                ["predicted_regime"],
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=14,
                    min_samples_leaf=4,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def apply_regime_aware_correction(
    residual_model: Pipeline,
    X: pd.DataFrame,
    predicted_regimes,
) -> np.ndarray:
    """Predict residuals from meteorological features + predicted regime and clamp >= 0."""
    feature_frame = add_regime_feature(X, predicted_regimes)
    predicted_residual = residual_model.predict(feature_frame)
    raw_forecast = X["rainfall_forecast"].to_numpy(dtype=float)
    return np.maximum(0.0, raw_forecast + predicted_residual)
