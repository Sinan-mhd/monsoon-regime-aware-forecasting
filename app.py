"""Streamlit interactive dashboard for Regime-Aware Monsoon Rainfall Forecasting."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.bias_correction import apply_regime_aware_correction, uniform_correct
from src.data_preprocessing import FEATURES

BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"
METRICS_PATH = REPORTS_DIR / "metrics.json"
MODEL_BUNDLE_PATH = MODELS_DIR / "monsoon_models.joblib"
PREDICTIONS_CSV_PATH = REPORTS_DIR / "test_predictions.csv"

PRESET_SCENARIOS = {
    "Custom Input": None,
    "Active Monsoon Example": {
        "rainfall_forecast": 48.0,
        "temperature_anomaly": -0.6,
        "humidity": 86.0,
        "wind_speed": 9.4,
        "pressure_anomaly": -1.6,
        "rainfall_lag_1d": 42.0,
        "rainfall_lag_3d": 45.0,
    },
    "Break Monsoon Example": {
        "rainfall_forecast": 18.0,
        "temperature_anomaly": 1.4,
        "humidity": 61.0,
        "wind_speed": 3.5,
        "pressure_anomaly": 1.3,
        "rainfall_lag_1d": 4.0,
        "rainfall_lag_3d": 5.5,
    },
    "Other / Transition Example": {
        "rainfall_forecast": 24.0,
        "temperature_anomaly": 0.1,
        "humidity": 74.0,
        "wind_speed": 6.1,
        "pressure_anomaly": 0.0,
        "rainfall_lag_1d": 14.0,
        "rainfall_lag_3d": 16.0,
    },
}


st.set_page_config(
    page_title="Regime-Aware Monsoon Rainfall Forecasting (SIH26080)",
    page_icon="🌧️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .banner-box {
        padding: 0.9rem 1.1rem;
        border-radius: 0.6rem;
        border-left: 5px solid #0284c7;
        background-color: rgba(2, 132, 199, 0.08);
        margin-bottom: 1rem;
    }
    .arch-card {
        padding: 0.85rem 1rem;
        border-radius: 0.55rem;
        border: 1px solid rgba(148, 163, 184, 0.35);
        background-color: rgba(15, 23, 42, 0.03);
        height: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_bundle(bundle_path: Path):
    return joblib.load(bundle_path)


def load_metrics(metrics_path: Path) -> dict:
    return json.loads(metrics_path.read_text(encoding="utf-8"))


st.title("Regime-Aware Post-Processing of Monsoon Rainfall Forecasts")
st.caption(
    "Problem Statement ID: SIH26080 | Organisation: Ministry of Earth Sciences (MoES) | "
    "Two-Stage ML Post-Processing & Evaluation Dashboard"
)

if not METRICS_PATH.exists() or not MODEL_BUNDLE_PATH.exists():
    st.warning(
        "Trained model artifacts or evaluation reports were not found. "
        "Run `python train.py` first to generate models and metrics."
    )
    if st.button("Run Training Pipeline Now (Synthetic Demo Data)"):
        from train import run_training_pipeline

        with st.spinner("Training regime classifier and bias-correction models..."):
            run_training_pipeline()
        st.success("Training completed. Reloading dashboard...")
        st.rerun()
    st.stop()

metrics = load_metrics(METRICS_PATH)
bundle = load_model_bundle(MODEL_BUNDLE_PATH)
dataset_info = metrics["dataset"]
is_synthetic = bool(dataset_info.get("is_synthetic_demo", True))

if is_synthetic:
    st.warning(
        "**Synthetic Demonstration Data Notice:** Current metrics and trained models were "
        "generated from a synthetic demonstration dataset (`data/demo_monsoon.csv`) to verify "
        "pipeline execution. Replace with quality-controlled historical forecast/observation "
        "records (`data/monsoon_data.csv`) before drawing scientific or operational conclusions."
    )
else:
    st.info(
        f"**Dataset Source:** `{dataset_info['source_path']}` — "
        f"{dataset_info.get('dataset_note', '')}"
    )

# Top KPI Strip
fc = metrics["forecast_comparison"]
skills = metrics["skill_scores_vs_raw"]
cls_metrics = metrics["regime_classification"]

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Regime Accuracy", f"{cls_metrics['accuracy']:.3f}", f"Macro F1: {cls_metrics['f1_macro']:.3f}")
k2.metric("Raw MAE", f"{fc['raw_forecast']['MAE']:.2f} mm/day", f"RMSE: {fc['raw_forecast']['RMSE']:.2f}")
k3.metric(
    "Uniform MAE",
    f"{fc['uniform_bias_correction']['MAE']:.2f} mm/day",
    f"Skill: {skills['uniform_bias_correction']['MAE_skill_score']:+.3f}",
)
k4.metric(
    "Regime-Aware MAE",
    f"{fc['regime_aware_correction']['MAE']:.2f} mm/day",
    f"Skill: {skills['regime_aware_correction']['MAE_skill_score']:+.3f}",
)
k5.metric(
    "Regime-Aware RMSE",
    f"{fc['regime_aware_correction']['RMSE']:.2f} mm/day",
    f"Skill: {skills['regime_aware_correction']['RMSE_skill_score']:+.3f}",
)

tabs = st.tabs(
    [
        "Overview & Architecture",
        "Dataset & Regime Classification",
        "Bias Correction Comparison",
        "Visualisations",
        "Sample Prediction Interface",
    ]
)

# ---------------------------------------------------------------------------
# Tab 1: Overview & Architecture
# ---------------------------------------------------------------------------
with tabs[0]:
    st.subheader("1. Project Overview")
    st.markdown(
        """
        <div class="banner-box">
        <strong>Core Motivation (SIH26080 — MoES):</strong> Raw numerical weather prediction (NWP)
        rainfall forecasts frequently exhibit systematic errors that reverse or shift in magnitude
        between <strong>Active</strong> (enhanced convection/trough) and <strong>Break</strong>
        (suppressed rainfall) monsoon phases. Applying a single uniform bias offset across all days
        can cancel opposing errors on average while degrading forecasts in individual regimes.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        **Implemented Objectives:**
        1. **Monsoon Regime Identification:** Classify forecast days into `Active`, `Break`, or `Other`
           using meteorological predictors available at forecast issuance time.
        2. **Baseline vs Proposed Post-Processing:** Compare unadjusted **Raw Forecasts** against
           **Uniform Bias Correction** (training-mean residual offset) and **Regime-Aware Residual Correction**.
        3. **Multi-Metric & Regime-Wise Evaluation:** Report overall and phase-specific **MAE**, **RMSE**,
           and **Skill Scores** relative to raw forecasts, plus full classification diagnostics.
        """
    )

    st.subheader("2. Model Architecture & Leakage Prevention")
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        st.markdown(
            """
            <div class="arch-card">
            <h4>Stage 0: Preprocessing</h4>
            <ul>
              <li>Schema &amp; type validation</li>
              <li>Domain-bound checks</li>
              <li>Chronological split when dates exist</li>
              <li>Training-only median imputation</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with a2:
        st.markdown(
            """
            <div class="arch-card">
            <h4>Stage 1: Regime Classifier</h4>
            <ul>
              <li>Random Forest Classifier</li>
              <li>Balanced class weights</li>
              <li>Predicts Active / Break / Other</li>
              <li>Out-of-fold train predictions</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with a3:
        st.markdown(
            """
            <div class="arch-card">
            <h4>Stage 2A: Uniform Baseline</h4>
            <ul>
              <li>Mean residual from training set only</li>
              <li><code>y_train - forecast_train</code></li>
              <li>Uniform additive offset</li>
              <li>Non-negative clamping (&ge; 0)</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with a4:
        st.markdown(
            """
            <div class="arch-card">
            <h4>Stage 2B: Regime-Aware</h4>
            <ul>
              <li>Random Forest Regressor</li>
              <li>Uses predictors + predicted regime</li>
              <li>Learns conditional residuals</li>
              <li>Non-negative clamping (&ge; 0)</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.success(
        "**Leakage Safeguards:** Imputers, one-hot encoders, uniform residual mean, regime classifier, "
        "and residual regressor are all fitted exclusively on training folds. Neither true test regimes "
        "nor observed test rainfall are ever supplied during test-set prediction."
    )
    st.caption(metrics["interpretation"])

# ---------------------------------------------------------------------------
# Tab 2: Dataset & Regime Classification
# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("3. Dataset Information")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Rows After Cleaning", f"{dataset_info['rows_after_cleaning']:,}")
    d2.metric("Training Rows", f"{dataset_info['train_rows']:,}")
    d3.metric("Test Rows", f"{dataset_info['test_rows']:,}")
    d4.metric("Split Strategy", dataset_info["split_strategy"].replace("_", " ").title())

    col_left, col_right = st.columns([1.2, 1])
    with col_left:
        st.markdown("**Regime Class Distribution (Cleaned Dataset)**")
        dist_df = pd.DataFrame(
            [
                {"Regime": k, "Samples": v}
                for k, v in dataset_info.get("regime_distribution", {}).items()
            ]
        )
        st.dataframe(dist_df, use_container_width=True, hide_index=True)

        cleaning = dataset_info.get("cleaning_report", {})
        if cleaning:
            st.markdown("**Data Cleaning Summary**")
            st.json(cleaning)

    with col_right:
        st.markdown("**Feature & Target Schema**")
        schema_rows = [
            ("rainfall_forecast", "Raw NWP rainfall forecast (mm/day)", "Predictor"),
            ("temperature_anomaly", "Temperature anomaly (°C)", "Predictor"),
            ("humidity", "Relative humidity (%)", "Predictor"),
            ("wind_speed", "Wind speed (m/s)", "Predictor"),
            ("pressure_anomaly", "Surface pressure anomaly (hPa)", "Predictor"),
            ("rainfall_lag_1d", "Previous day's rainfall (mm/day)", "Predictor"),
            ("rainfall_lag_3d", "Previous 3 days' rainfall (mm/day)", "Predictor"),
            ("observed_rainfall", "Observed rainfall (mm/day)", "Target"),
            ("regime", "Monsoon phase (Active / Break / Other)", "Classification Target"),
        ]
        st.dataframe(
            pd.DataFrame(schema_rows, columns=["Column", "Description", "Role"]),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    st.subheader("4. Regime Classification Results (Test Set)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{cls_metrics['accuracy']:.3f}")
    c2.metric("Macro Precision", f"{cls_metrics['precision_macro']:.3f}")
    c3.metric("Macro Recall", f"{cls_metrics['recall_macro']:.3f}")
    c4.metric("Macro F1-Score", f"{cls_metrics['f1_macro']:.3f}")

    cm_col1, cm_col2 = st.columns([1.15, 1])
    with cm_col1:
        st.markdown("**Per-Class Classification Metrics**")
        per_cls_rows = [
            {
                "Regime": label,
                "Precision": round(vals["precision"], 4),
                "Recall": round(vals["recall"], 4),
                "F1-Score": round(vals["f1_score"], 4),
                "Support": vals["support"],
            }
            for label, vals in cls_metrics["per_class"].items()
        ]
        st.dataframe(pd.DataFrame(per_cls_rows), use_container_width=True, hide_index=True)
    with cm_col2:
        cm_img = REPORTS_DIR / "confusion_matrix.png"
        if cm_img.exists():
            st.image(str(cm_img), caption="Confusion Matrix (Test Set)", use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 3: Bias Correction Comparison & Regime-Specific Performance
# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("5. Overall Approach Comparison & Skill Scores")
    st.markdown(
        "**Skill Score Definition:** "
        "`Skill Score = 1 - (Corrected Model Error / Raw Forecast Error)`. "
        "Positive values indicate error reduction relative to raw forecasts; negative values indicate degradation."
    )

    overall_rows = [
        {
            "Approach": "Raw Rainfall Forecast",
            "MAE (mm/day)": round(fc["raw_forecast"]["MAE"], 3),
            "RMSE (mm/day)": round(fc["raw_forecast"]["RMSE"], 3),
            "MAE Skill Score": 0.0,
            "RMSE Skill Score": 0.0,
        },
        {
            "Approach": "Uniform Bias Correction (Baseline)",
            "MAE (mm/day)": round(fc["uniform_bias_correction"]["MAE"], 3),
            "RMSE (mm/day)": round(fc["uniform_bias_correction"]["RMSE"], 3),
            "MAE Skill Score": round(skills["uniform_bias_correction"]["MAE_skill_score"], 4),
            "RMSE Skill Score": round(skills["uniform_bias_correction"]["RMSE_skill_score"], 4),
        },
        {
            "Approach": "Regime-Aware Bias Correction (Proposed)",
            "MAE (mm/day)": round(fc["regime_aware_correction"]["MAE"], 3),
            "RMSE (mm/day)": round(fc["regime_aware_correction"]["RMSE"], 3),
            "MAE Skill Score": round(skills["regime_aware_correction"]["MAE_skill_score"], 4),
            "RMSE Skill Score": round(skills["regime_aware_correction"]["RMSE_skill_score"], 4),
        },
    ]
    st.dataframe(pd.DataFrame(overall_rows), use_container_width=True, hide_index=True)

    st.caption(
        f"Uniform baseline additive offset learned on training set: "
        f"`{metrics['training_summary']['uniform_mean_training_residual_mm_day']:+.3f} mm/day`"
    )

    st.subheader("6. Performance by Monsoon Regime (Active / Break / Other)")
    regime_perf = metrics["regime_specific_performance"]
    regime_rows = []
    for regime_name, rdata in regime_perf.items():
        m = rdata["methods"]
        sk = rdata["skill_scores_vs_raw"]
        regime_rows.append(
            {
                "Regime": regime_name,
                "Test Samples": rdata["support"],
                "Raw MAE": round(m["raw_forecast"]["MAE"], 3),
                "Uniform MAE": round(m["uniform_bias_correction"]["MAE"], 3),
                "Regime-Aware MAE": round(m["regime_aware_correction"]["MAE"], 3),
                "Uniform MAE Skill": round(sk["uniform_bias_correction"]["MAE_skill_score"], 4),
                "Regime-Aware MAE Skill": round(sk["regime_aware_correction"]["MAE_skill_score"], 4),
                "Raw RMSE": round(m["raw_forecast"]["RMSE"], 3),
                "Uniform RMSE": round(m["uniform_bias_correction"]["RMSE"], 3),
                "Regime-Aware RMSE": round(m["regime_aware_correction"]["RMSE"], 3),
                "Uniform RMSE Skill": round(sk["uniform_bias_correction"]["RMSE_skill_score"], 4),
                "Regime-Aware RMSE Skill": round(sk["regime_aware_correction"]["RMSE_skill_score"], 4),
            }
        )
    st.dataframe(pd.DataFrame(regime_rows), use_container_width=True, hide_index=True)

    if PREDICTIONS_CSV_PATH.exists():
        st.subheader("Sample Test-Set Predictions")
        preds_df = pd.read_csv(PREDICTIONS_CSV_PATH)
        regime_filter = st.multiselect(
            "Filter by true monsoon regime",
            options=sorted(preds_df["true_regime"].unique()),
            default=sorted(preds_df["true_regime"].unique()),
        )
        filtered = preds_df[preds_df["true_regime"].isin(regime_filter)]
        st.dataframe(filtered.head(30), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Tab 4: Visualisations
# ---------------------------------------------------------------------------
with tabs[3]:
    st.subheader("7. Generated Evaluation Visualisations")
    v1, v2 = st.columns(2)
    with v1:
        mae_img = REPORTS_DIR / "mae_comparison.png"
        if mae_img.exists():
            st.image(str(mae_img), caption="Overall MAE Comparison", use_container_width=True)
    with v2:
        rmse_img = REPORTS_DIR / "rmse_comparison.png"
        if rmse_img.exists():
            st.image(str(rmse_img), caption="Overall RMSE Comparison", use_container_width=True)

    reg_img = REPORTS_DIR / "regime_performance.png"
    if reg_img.exists():
        st.image(str(reg_img), caption="MAE and RMSE Across Monsoon Regimes", use_container_width=True)

    avp_img = REPORTS_DIR / "actual_vs_predicted.png"
    if avp_img.exists():
        st.image(str(avp_img), caption="Actual vs Predicted Rainfall (Test Set)", use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 5: Interactive Sample Rainfall Prediction
# ---------------------------------------------------------------------------
with tabs[4]:
    st.subheader("8. Sample Rainfall Post-Processing Predictor")
    st.write(
        "Enter meteorological forecast inputs to predict the monsoon regime and compare raw, "
        "uniformly corrected, and regime-aware corrected rainfall forecasts."
    )

    preset_choice = st.selectbox(
        "Load an illustrative preset scenario (optional)",
        options=list(PRESET_SCENARIOS.keys()),
        index=1,
    )
    defaults = PRESET_SCENARIOS[preset_choice] or {
        "rainfall_forecast": 25.0,
        "temperature_anomaly": 0.0,
        "humidity": 75.0,
        "wind_speed": 6.5,
        "pressure_anomaly": 0.0,
        "rainfall_lag_1d": 15.0,
        "rainfall_lag_3d": 18.0,
    }

    with st.form("prediction_form"):
        in1, in2, in3 = st.columns(3)
        with in1:
            rainfall_forecast = st.number_input(
                "Raw Rainfall Forecast (mm/day)", min_value=0.0, max_value=500.0,
                value=float(defaults["rainfall_forecast"]), step=0.5
            )
            temperature_anomaly = st.number_input(
                "Temperature Anomaly (°C)", min_value=-10.0, max_value=10.0,
                value=float(defaults["temperature_anomaly"]), step=0.1
            )
            humidity = st.slider(
                "Relative Humidity (%)", min_value=0.0, max_value=100.0,
                value=float(defaults["humidity"]), step=0.5
            )
        with in2:
            wind_speed = st.number_input(
                "Wind Speed (m/s)", min_value=0.0, max_value=60.0,
                value=float(defaults["wind_speed"]), step=0.2
            )
            pressure_anomaly = st.number_input(
                "Pressure Anomaly (hPa)", min_value=-20.0, max_value=20.0,
                value=float(defaults["pressure_anomaly"]), step=0.1
            )
        with in3:
            rainfall_lag_1d = st.number_input(
                "Previous Day's Rainfall (mm/day)", min_value=0.0, max_value=500.0,
                value=float(defaults["rainfall_lag_1d"]), step=0.5
            )
            rainfall_lag_3d = st.number_input(
                "Previous 3 Days' Rainfall (mm/day)", min_value=0.0, max_value=500.0,
                value=float(defaults["rainfall_lag_3d"]), step=0.5
            )

        submitted = st.form_submit_button("Run Regime-Aware Post-Processing", type="primary")

    if submitted or preset_choice != "Custom Input":
        input_row = pd.DataFrame(
            [
                {
                    "rainfall_forecast": rainfall_forecast,
                    "temperature_anomaly": temperature_anomaly,
                    "humidity": humidity,
                    "wind_speed": wind_speed,
                    "pressure_anomaly": pressure_anomaly,
                    "rainfall_lag_1d": rainfall_lag_1d,
                    "rainfall_lag_3d": rainfall_lag_3d,
                }
            ],
            columns=FEATURES,
        )

        classifier = bundle["classifier"]
        residual_model = bundle["residual_model"]
        uniform_offset = float(bundle["uniform_residual"])

        pred_regime = str(classifier.predict(input_row)[0])
        proba = classifier.predict_proba(input_row)[0]
        classes = [str(c) for c in classifier.classes_]

        uniform_val = float(uniform_correct([rainfall_forecast], uniform_offset)[0])
        regime_aware_val = float(
            apply_regime_aware_correction(residual_model, input_row, [pred_regime])[0]
        )

        st.markdown("### Prediction Output")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Predicted Monsoon Regime", pred_regime)
        r2.metric("Raw Rainfall Forecast", f"{rainfall_forecast:.2f} mm/day")
        r3.metric(
            "Uniformly Corrected",
            f"{uniform_val:.2f} mm/day",
            f"{uniform_val - rainfall_forecast:+.2f} mm/day",
        )
        r4.metric(
            "Regime-Aware Corrected",
            f"{regime_aware_val:.2f} mm/day",
            f"{regime_aware_val - rainfall_forecast:+.2f} mm/day",
        )

        proba_df = pd.DataFrame(
            {"Regime": classes, "Predicted Probability": [round(float(p), 4) for p in proba]}
        ).sort_values("Predicted Probability", ascending=False)
        st.markdown("**Regime Classifier Confidence**")
        st.dataframe(proba_df, use_container_width=True, hide_index=True)
