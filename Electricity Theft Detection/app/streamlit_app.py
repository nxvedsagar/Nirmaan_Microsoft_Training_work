"""
Streamlit Demo App - Electricity Theft Detection in Distribution Networks
B.Tech EEE Capstone Project

Run with:
    streamlit run streamlit_app.py
"""

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Electricity Theft Detection", page_icon="⚡", layout="wide")

MODEL_PATH = "/Users/nivedsagark/Documents/Projects/Electricity Theft Detection/model/theft_detection_model.pkl"
SCALER_PATH = "/Users/nivedsagark/Documents/Projects/Electricity Theft Detection/model/scaler.pkl"
FEATURES_PATH = "/Users/nivedsagark/Documents/Projects/Electricity Theft Detection/model/feature_columns.json"


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_cols = joblib.load(FEATURES_PATH)
    return model, scaler, feature_cols


MODEL_NEEDS_SCALING = False  # Gradient Boosting / Random Forest don't need scaling; set True for LR/SVM


def extract_features_from_series(series: np.ndarray) -> dict:
    n_days = len(series)
    first_half = series[: n_days // 2]
    second_half = series[n_days // 2:]

    mean_val = series.mean()
    std_val = series.std()
    cv = std_val / (mean_val + 1e-6)
    zero_frac = np.mean(series < 0.15)
    trend_ratio = (second_half.mean() + 1e-6) / (first_half.mean() + 1e-6)

    weekday_vals = series[np.arange(n_days) % 7 < 5]
    weekend_vals = series[np.arange(n_days) % 7 >= 5]
    weekday_weekend_ratio = (weekend_vals.mean() + 1e-6) / (weekday_vals.mean() + 1e-6)

    diffs = np.diff(series)
    sudden_drop_count = int(np.sum(diffs < -0.5 * (mean_val + 1e-6)))

    n_weeks = n_days // 7
    weekly_totals = series[: n_weeks * 7].reshape(n_weeks, 7).sum(axis=1) if n_weeks > 0 else np.array([series.sum()])
    weekly_cv = weekly_totals.std() / (weekly_totals.mean() + 1e-6)

    p40 = np.percentile(series, 40)
    near_cap_fraction = np.mean(np.abs(series - p40) < 0.05 * (mean_val + 1e-6))

    return {
        "mean_daily_kwh": mean_val,
        "std_daily_kwh": std_val,
        "coefficient_of_variation": cv,
        "min_daily_kwh": series.min(),
        "max_daily_kwh": series.max(),
        "median_daily_kwh": np.median(series),
        "zero_day_fraction": zero_frac,
        "trend_ratio": trend_ratio,
        "weekday_weekend_ratio": weekday_weekend_ratio,
        "sudden_drop_count": sudden_drop_count,
        "weekly_cv": weekly_cv,
        "near_cap_fraction": near_cap_fraction,
        "max_to_mean_ratio": series.max() / (mean_val + 1e-6),
    }


def predict_one(series, model, scaler, feature_cols):
    feats = extract_features_from_series(series)
    X = pd.DataFrame([feats])[feature_cols]
    X_input = scaler.transform(X) if MODEL_NEEDS_SCALING else X
    proba = model.predict_proba(X_input)[0, 1]
    pred = "Theft" if proba >= 0.5 else "Normal"
    return pred, proba, feats


# ---------------------------------------------------------------- UI -----

st.title("⚡ Electricity Theft Detection in Distribution Networks")
st.caption("B.Tech EEE Capstone Project · AI/ML-based smart-meter anomaly detection")

with st.expander("ℹ️ About this project", expanded=False):
    st.markdown("""
    This tool analyses a consumer's **daily electricity consumption pattern**
    (from smart-meter / AMI data) and flags a **theft / meter-tampering risk
    score** using a trained Gradient Boosting classifier (97.2% accuracy,
    0.89 F1-score on the theft class, 0.99 ROC-AUC on held-out test data).

    It is intended to help utilities **prioritise field inspections** rather
    than checking meters at random.
    """)

try:
    model, scaler, feature_cols = load_artifacts()
except FileNotFoundError:
    st.error("Model artifacts not found. Please run the training notebook first "
             "to generate `theft_detection_model.pkl`, `scaler.pkl`, and `feature_columns.json`.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["🔍 Single Consumer Check", "📁 Batch Upload (CSV)", "📊 Model Performance"])

# ---------------- TAB 1: single consumer, manual/simulated series ----------------
with tab1:
    st.subheader("Check a single consumer")
    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown("**Option A — Simulate a consumption pattern**")
        pattern = st.selectbox(
            "Pattern type (for demo/testing)",
            ["Normal Consumer", "Sudden Drop (tampering)", "Zero Stretches (bypass)",
             "Flattened / Capped Reading", "Gradual Under-reporting", "Periodic Weekend Bypass"]
        )
        base_load = st.slider("Average base load (kWh/day)", 2.0, 25.0, 10.0)
        n_days = st.slider("Number of days of data", 30, 180, 90, step=30)
        simulate_btn = st.button("Generate & Predict", type="primary")

    with col_b:
        st.markdown("**Option B — Paste your own daily readings**")
        manual_input = st.text_area(
            "Comma-separated daily kWh values",
            placeholder="10.2, 9.8, 10.5, 3.1, 3.0, 2.9, 10.1, ...",
            height=120
        )
        manual_btn = st.button("Predict from pasted data")

    series = None
    if simulate_btn:
        rng = np.random.default_rng()
        t = np.arange(n_days)
        seasonal = 2 * np.sin(2 * np.pi * t / 365 + rng.uniform(0, 6.28))
        weekly = 1 * np.sin(2 * np.pi * t / 7 + rng.uniform(0, 6.28))
        noise = rng.normal(0, 1, n_days)
        series = np.clip(base_load + seasonal + weekly + noise, 0, None)

        if pattern == "Sudden Drop (tampering)":
            start = rng.integers(int(n_days * 0.3), int(n_days * 0.6))
            series[start:] *= rng.uniform(0.15, 0.35)
        elif pattern == "Zero Stretches (bypass)":
            for _ in range(rng.integers(3, 8)):
                s = rng.integers(0, n_days - 10)
                series[s:s + rng.integers(2, 6)] = rng.uniform(0, 0.2)
        elif pattern == "Flattened / Capped Reading":
            cap = np.percentile(series, 40)
            series = np.minimum(series, cap) * rng.uniform(0.8, 1.0)
        elif pattern == "Gradual Under-reporting":
            decay = np.linspace(1.0, rng.uniform(0.25, 0.45), n_days)
            series = series * decay
        elif pattern == "Periodic Weekend Bypass":
            mask = (np.arange(n_days) % 7 >= 5)
            series[mask] *= rng.uniform(0.1, 0.3)

    if manual_btn and manual_input.strip():
        try:
            series = np.array([float(x.strip()) for x in manual_input.split(",") if x.strip() != ""])
            if len(series) < 14:
                st.warning("Please provide at least 14 daily readings for a reliable prediction.")
                series = None
        except ValueError:
            st.error("Could not parse the input. Please provide comma-separated numbers only.")

    if series is not None:
        pred, proba, feats = predict_one(series, model, scaler, feature_cols)

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=series, mode="lines", name="Daily kWh",
                                  line=dict(color="firebrick" if pred == "Theft" else "seagreen")))
        fig.update_layout(title="Daily Consumption Pattern", xaxis_title="Day",
                           yaxis_title="kWh", height=350)
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Prediction", pred)
        c2.metric("Theft Probability", f"{proba*100:.1f}%")
        c3.metric("Consistency (CV)", f"{feats['coefficient_of_variation']:.2f}")

        if pred == "Theft":
            st.error(f"⚠️ High theft/tampering risk detected ({proba*100:.1f}% probability). "
                     "Recommend prioritising this consumer for field inspection.")
        else:
            st.success(f"✅ Consumption pattern appears normal ({(1-proba)*100:.1f}% confidence).")

        with st.expander("View engineered features used by the model"):
            st.dataframe(pd.DataFrame([feats]).T.rename(columns={0: "Value"}))

# ---------------- TAB 2: batch CSV upload ----------------
with tab2:
    st.subheader("Batch prediction from CSV")
    st.markdown("""
    Upload a CSV where **each row is one consumer** and **each column (after
    `consumer_id`) is one day's kWh reading** — the same format as the raw
    training dataset (`smart_meter_data.csv`).
    """)
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        id_col = "consumer_id" if "consumer_id" in batch_df.columns else batch_df.columns[0]
        exclude = {id_col, "theft_type", "label", "Label"}
        day_cols = [c for c in batch_df.columns if c not in exclude]

        results = []
        for _, row in batch_df.iterrows():
            series = row[day_cols].astype(float).interpolate(limit_direction="both").fillna(0).values
            pred, proba, _ = predict_one(series, model, scaler, feature_cols)
            results.append({"consumer_id": row[id_col], "Prediction": pred, "Theft_Probability": round(proba, 4)})

        result_df = pd.DataFrame(results).sort_values("Theft_Probability", ascending=False)
        st.dataframe(result_df, use_container_width=True)

        n_flagged = (result_df["Prediction"] == "Theft").sum()
        st.info(f"🔎 {n_flagged} out of {len(result_df)} consumers flagged as high theft-risk.")

        csv_out = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download results as CSV", csv_out, "theft_predictions.csv", "text/csv")

# ---------------- TAB 3: model performance summary ----------------
with tab3:
    st.subheader("Model performance (held-out test set)")
    try:
        perf_df = pd.read_csv("../report_assets/model_comparison.csv")
        st.dataframe(perf_df, use_container_width=True)
    except FileNotFoundError:
        st.warning("Model comparison results not found.")

    col1, col2 = st.columns(2)
    with col1:
        st.image("/Users/nivedsagark/Documents/Projects/Electricity Theft Detection/report_assets/confusion_matrix.png", caption="Confusion Matrix - Best Model")
    with col2:
        st.image("/Users/nivedsagark/Documents/Projects/Electricity Theft Detection/report_assets/roc_curve.png", caption="ROC Curve - Best Model")

    st.image("/Users/nivedsagark/Documents/Projects/Electricity Theft Detection/report_assets/feature_importance.png", caption="Feature Importance")

st.markdown("---")
st.caption("Capstone Project · Electricity Theft Detection using Machine Learning · Department of EEE")
