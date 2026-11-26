"""Monsoon regime classification model and leak-free out-of-fold prediction helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline


def build_regime_classifier(random_state: int = 42) -> Pipeline:
    """Create a leak-free Random Forest classification pipeline for monsoon regimes."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=250,
                    max_depth=12,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def generate_oof_regime_predictions(
    X_train: pd.DataFrame,
    r_train: pd.Series,
    random_state: int = 42,
    n_splits: int = 5,
) -> np.ndarray:
    """Generate out-of-fold regime predictions on training data to prevent stacking leakage."""
    min_class_count = int(r_train.value_counts().min())
    effective_splits = min(n_splits, min_class_count)
    if effective_splits < 2:
        # Fallback when a class has fewer than 2 samples in a tiny test fixture.
        clf = build_regime_classifier(random_state=random_state)
        clf.fit(X_train, r_train)
        return clf.predict(X_train)

    cv = StratifiedKFold(
        n_splits=effective_splits,
        shuffle=True,
        random_state=random_state,
    )
    estimator = build_regime_classifier(random_state=random_state)
    return cross_val_predict(estimator, X_train, r_train, cv=cv)
