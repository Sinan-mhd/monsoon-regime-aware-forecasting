"""Data loading, synthetic demo generation, cleaning, and splitting utilities."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES = [
    "rainfall_forecast",
    "temperature_anomaly",
    "humidity",
    "wind_speed",
    "pressure_anomaly",
    "rainfall_lag_1d",
    "rainfall_lag_3d",
]
TARGET = "observed_rainfall"
REGIME = "regime"
DATE_COL = "date"
VALID_REGIMES = ("Active", "Break", "Other")

DEFAULT_REAL_DATA_PATH = Path("data") / "monsoon_data.csv"
DEFAULT_DEMO_DATA_PATH = Path("data") / "demo_monsoon.csv"

_REGIME_CANONICAL_MAP = {r.lower(): r for r in VALID_REGIMES}


def make_demo_data(
    n_samples: int = 2400,
    random_state: int = 42,
    include_dates: bool = False,
) -> pd.DataFrame:
    """Create clearly labelled synthetic demonstration data for end-to-end testing.

    Note:
        This generator produces illustrative synthetic patterns with regime-dependent
        bias solely to demonstrate and verify the software pipeline. It is NOT a
        physical meteorological simulation and must not be used for scientific claims.
    """
    rng = np.random.default_rng(random_state)
    regimes = rng.choice(list(VALID_REGIMES), size=n_samples, p=[0.35, 0.30, 0.35])

    temp = rng.normal(0.0, 1.2, n_samples)
    humidity = np.clip(
        np.where(
            regimes == "Active",
            rng.normal(84.0, 5.0, n_samples),
            np.where(
                regimes == "Break",
                rng.normal(64.0, 6.0, n_samples),
                rng.normal(74.0, 6.5, n_samples),
            ),
        ),
        20.0,
        100.0,
    )
    wind = np.maximum(
        0.2,
        np.where(
            regimes == "Active",
            rng.normal(9.0, 1.6, n_samples),
            np.where(
                regimes == "Break",
                rng.normal(3.8, 1.1, n_samples),
                rng.normal(6.2, 1.5, n_samples),
            ),
        ),
    )
    pressure = np.where(
        regimes == "Active",
        rng.normal(-1.4, 0.65, n_samples),
        np.where(
            regimes == "Break",
            rng.normal(1.2, 0.65, n_samples),
            rng.normal(0.0, 0.75, n_samples),
        ),
    )
    lag1 = np.maximum(
        0.0,
        np.where(
            regimes == "Active",
            rng.gamma(3.2, 11.5, n_samples),
            np.where(
                regimes == "Break",
                rng.gamma(1.1, 4.5, n_samples),
                rng.gamma(1.8, 7.5, n_samples),
            ),
        ),
    )
    lag3 = np.maximum(
        0.0,
        lag1 * rng.normal(1.05, 0.3, n_samples) + rng.normal(0.0, 6.0, n_samples),
    )

    base = np.where(regimes == "Active", 44.0, np.where(regimes == "Break", 7.0, 20.0))
    observed = np.maximum(
        0.0,
        base
        + 0.32 * lag1
        + 0.15 * (humidity - 70.0)
        + 1.3 * wind
        - 2.2 * pressure
        - 0.8 * temp
        + rng.normal(0.0, 8.5, n_samples),
    )

    # Deliberately regime-dependent systematic forecast error for pipeline demonstration:
    # Active phase under-forecasts convective rainfall; Break phase over-forecasts rainfall.
    forecast_bias = np.where(
        regimes == "Active",
        -12.5 - 0.10 * wind - 0.05 * lag1,
        np.where(
            regimes == "Break",
            9.5 + 0.6 * pressure,
            -1.0,
        ),
    )
    forecast = np.maximum(0.0, observed + forecast_bias + rng.normal(0.0, 9.0, n_samples))

    df = pd.DataFrame(
        {
            "rainfall_forecast": np.round(forecast, 3),
            "temperature_anomaly": np.round(temp, 3),
            "humidity": np.round(humidity, 3),
            "wind_speed": np.round(wind, 3),
            "pressure_anomaly": np.round(pressure, 3),
            "rainfall_lag_1d": np.round(lag1, 3),
            "rainfall_lag_3d": np.round(lag3, 3),
            "observed_rainfall": np.round(observed, 3),
            "regime": regimes,
        }
    )

    if include_dates:
        df.insert(0, DATE_COL, pd.date_range("2020-06-01", periods=n_samples, freq="D"))

    return df


def resolve_data_path(data_path: str | Path | None = None) -> tuple[Path, bool]:
    """Resolve dataset path, creating synthetic demonstration data when no dataset exists.

    Returns:
        Tuple of (resolved_path, is_synthetic_demo).
    """
    if data_path is not None:
        return Path(data_path), False

    if DEFAULT_REAL_DATA_PATH.exists():
        return DEFAULT_REAL_DATA_PATH, False

    DEFAULT_DEMO_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    demo_df = make_demo_data()
    demo_df.to_csv(DEFAULT_DEMO_DATA_PATH, index=False)
    return DEFAULT_DEMO_DATA_PATH, True


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load dataset from CSV, generating synthetic demo data only for default fallback."""
    if path is None:
        resolved_path, _ = resolve_data_path(None)
        return pd.read_csv(resolved_path)

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError(f"Dataset is empty: {file_path}")
    return df


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate schema, coerce types, sanitize invalid values, and preserve predictor NaNs for training-only imputation.

    Rows missing target (`observed_rainfall`), `rainfall_forecast`, or valid `regime`
    labels are removed because they cannot be reliably trained or evaluated. Invalid
    physical bounds in auxiliary meteorological predictors (e.g., negative wind speed or
    humidity outside [0, 100]) are coerced to NaN so they are imputed strictly from
    training-set statistics inside the scikit-learn pipeline.
    """
    if df is None or df.empty:
        raise ValueError("Input dataframe is empty.")

    raw_rows = int(len(df))
    required = FEATURES + [TARGET, REGIME]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    has_date = DATE_COL in df.columns
    cols_to_keep = ([DATE_COL] if has_date else []) + required
    clean = df[cols_to_keep].copy()

    if has_date:
        clean[DATE_COL] = pd.to_datetime(clean[DATE_COL], errors="coerce")
        clean = clean.dropna(subset=[DATE_COL])

    for col in FEATURES + [TARGET]:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")

    clean = clean.replace([np.inf, -np.inf], np.nan)

    # Standardize regime strings and reject invalid labels.
    regime_series = clean[REGIME].astype("string").str.strip()
    clean[REGIME] = regime_series.str.lower().map(_REGIME_CANONICAL_MAP)

    # Drop rows with missing or negative target / raw forecast, or invalid regime.
    clean = clean.dropna(subset=["rainfall_forecast", TARGET, REGIME])
    clean = clean[(clean["rainfall_forecast"] >= 0.0) & (clean[TARGET] >= 0.0)]

    # Coerce physically impossible predictor measurements to NaN for training-fold imputation.
    invalid_predictor_cells = 0
    for non_neg_col in ("wind_speed", "rainfall_lag_1d", "rainfall_lag_3d"):
        invalid_mask = clean[non_neg_col] < 0.0
        invalid_predictor_cells += int(invalid_mask.sum())
        clean.loc[invalid_mask, non_neg_col] = np.nan

    humidity_invalid = (clean["humidity"] < 0.0) | (clean["humidity"] > 100.0)
    invalid_predictor_cells += int(humidity_invalid.sum())
    clean.loc[humidity_invalid, "humidity"] = np.nan

    if has_date:
        clean = clean.sort_values(DATE_COL).reset_index(drop=True)
    else:
        clean = clean.reset_index(drop=True)

    if clean[REGIME].nunique() < 2:
        raise ValueError("At least two valid monsoon regime classes are required.")
    if len(clean) < 50:
        raise ValueError("At least 50 valid rows are required after cleaning.")

    missing_feature_cells = int(clean[FEATURES].isna().sum().sum())
    clean.attrs["cleaning_report"] = {
        "raw_rows": raw_rows,
        "valid_rows": int(len(clean)),
        "dropped_rows": raw_rows - int(len(clean)),
        "invalid_predictor_values_coerced_to_nan": invalid_predictor_cells,
        "predictor_missing_cells_to_impute": missing_feature_cells,
        "has_date_column": has_date,
    }
    return clean


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    split_strategy: str = "auto",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series, str]:
    """Split dataset chronologically when dates exist, or via stratified random split otherwise."""
    if not 0.05 <= test_size <= 0.5:
        raise ValueError("test_size must be between 0.05 and 0.5.")

    has_date = DATE_COL in df.columns
    use_chronological = (
        split_strategy == "chronological"
        or (split_strategy == "auto" and has_date)
    )

    if use_chronological:
        if not has_date:
            raise ValueError("Chronological split requested, but 'date' column is not present.")
        ordered = df.sort_values(DATE_COL).reset_index(drop=True)
        split_idx = int(len(ordered) * (1.0 - test_size))
        if split_idx < 10 or (len(ordered) - split_idx) < 5:
            raise ValueError("Dataset is too small for the requested chronological split.")

        train_df = ordered.iloc[:split_idx]
        test_df = ordered.iloc[split_idx:]
        if train_df[REGIME].nunique() < 2:
            raise ValueError("Training portion of chronological split has fewer than 2 regimes.")

        return (
            train_df[FEATURES].copy(),
            test_df[FEATURES].copy(),
            train_df[TARGET].copy(),
            test_df[TARGET].copy(),
            train_df[REGIME].copy(),
            test_df[REGIME].copy(),
            "chronological",
        )

    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    regimes = df[REGIME].copy()

    X_train, X_test, y_train, y_test, r_train, r_test = train_test_split(
        X,
        y,
        regimes,
        test_size=test_size,
        random_state=random_state,
        stratify=regimes,
    )
    return X_train, X_test, y_train, y_test, r_train, r_test, "stratified_random"
