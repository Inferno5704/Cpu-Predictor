# app_streamlit.py
import os
import io
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics import mean_squared_error, r2_score
from mlflow.tracking import MlflowClient
import mlflow
import plotly.express as px

# -------- CONFIG ----------
st.set_page_config(page_title="CPU Usage Predictor", layout="wide", initial_sidebar_state="collapsed")
ARTIFACT_DIR = "artifacts"
ARTIFACT_MODEL = os.path.join(ARTIFACT_DIR, "best_model.pkl")
ARTIFACT_FI = os.path.join(ARTIFACT_DIR, "feature_importances.csv")
ARTIFACT_RUN_SUMMARY = os.path.join(ARTIFACT_DIR, "run_summary.csv")
EXPERIMENT_NAME = "cpu_usage_experiment"
REQUIRED_COLS = ['cpu_request', 'mem_request', 'cpu_limit', 'mem_limit', 'runtime_minutes', 'controller_kind']

# -------- COMPACT CSS ----------
st.markdown(
    """
    <style>
    .css-18e3th9 { padding: 0.6rem 1rem; }
    .block-container { padding-top: 0.4rem; padding-bottom: 0.6rem; }
    .stButton>button { padding: .45rem .6rem; }
    .card { background: white; border-radius:8px; padding:12px; box-shadow: 0 6px 20px rgba(0,0,0,0.04); }
    .small { font-size:13px; color:#6b7280; margin:0; padding:0; }
    .tight { margin-top:4px; margin-bottom:4px; }
    .mono { font-family: monospace; font-size:13px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------- HELPERS ----------
@st.cache_data
def load_local_model(path=ARTIFACT_MODEL):
    if os.path.exists(path):
        return joblib.load(path)
    return None

@st.cache_data
def read_csv_safe(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def fetch_mlflow_runs(experiment_name=EXPERIMENT_NAME, max_results=200):
    try:
        client = MlflowClient()
        exp = client.get_experiment_by_name(experiment_name)
        if not exp:
            return pd.DataFrame()
        runs = client.search_runs(exp.experiment_id, order_by=["metrics.rmse ASC"], max_results=max_results)
        rows = []
        for r in runs:
            rows.append({
                "run_id": r.info.run_id,
                "model": r.data.params.get("model"),
                "rmse": r.data.metrics.get("rmse"),
                "r2": r.data.metrics.get("r2"),
                "train_time_s": r.data.metrics.get("train_time_seconds")
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

def try_load_mlflow_model(run_id):
    try:
        model_uri = f"runs:/{run_id}/model"
        m = mlflow.sklearn.load_model(model_uri)
        return m
    except Exception:
        return None

def bytes_to_mb(x):
    try:
        return float(x) / (1024 ** 2)
    except Exception:
        return np.nan

# -------- HEADER ----------
col_h1, col_h2 = st.columns([3, 1], gap="small")
with col_h1:
    st.markdown("<h2 style='margin:0'>CPU Usage Predictor</h2>", unsafe_allow_html=True)
    st.markdown("<p class='small tight'>Compare model predictions — single-row and dataset mode.</p>", unsafe_allow_html=True)
with col_h2:
    if os.path.exists(ARTIFACT_MODEL):
        st.success("Local model found")
    else:
        st.warning("Local model missing")

st.markdown("")  # spacer

# -------- TOP ROW (controls + upload + single-predict inputs) ----------
ctrl_col, upload_col, predict_col = st.columns([1.2, 2.2, 1.6], gap="small")

with ctrl_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<b>Model controls</b>", unsafe_allow_html=True)
    load_local = st.button("Load local model")
    refresh_mlflow = st.button("Refresh MLflow runs")
    st.markdown(f"<div class='small tight mono'>Local model: {ARTIFACT_MODEL}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with upload_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<b>Upload dataset (optional)</b>", unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["csv"], key="upload", help="Required cols: cpu_request, mem_request, cpu_limit, mem_limit, runtime_minutes, controller_kind")
    if uploaded:
        df = read_csv_safe(uploaded)
        if df is None:
            st.error("Could not read CSV")
        else:
            st.success("Dataset loaded")
            st.dataframe(df.head(6), use_container_width=True)
    else:
        df = None
        st.markdown("<div class='small tight'>Upload a CSV to run predictions & compare models.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with predict_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<b>Single-row input</b>", unsafe_allow_html=True)
    c1, c2 = st.columns([1,1], gap="small")
    with c1:
        cpu_request = st.number_input("cpu_request", value=0.2, step=0.01, format="%.3f")
        cpu_limit = st.number_input("cpu_limit", value=0.5, step=0.01, format="%.3f")
        runtime_minutes = st.number_input("runtime_minutes", value=0.0, format="%.1f")
    with c2:
        mem_request = st.number_input("mem_request (bytes)", value=134217728, format="%d")
        mem_limit = st.number_input("mem_limit (bytes)", value=268435456, format="%d")
        controller_kind = st.selectbox("controller_kind", options=["Job", "ReplicaSet", "Deployment", "Unknown"])
    predict_button = st.button("Predict single")
    st.markdown("</div>", unsafe_allow_html=True)

# -------- MODEL LOAD ----------
model = None
if load_local:
    model = load_local_model()
    if model is None:
        st.error("Local model not found — run training first.")
    else:
        st.success("Local model loaded")

# auto-load if available
if model is None:
    model = load_local_model()

# -------- BODY (left: dataset/predictions, right: MLflow info) ----------
left, right = st.columns([2, 1], gap="small")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<b>Predictions & Model Comparison (dataset)</b>", unsafe_allow_html=True)

    if df is None:
        st.markdown("<div class='small tight'>Upload a dataset to compare models on many rows. Single-row compare is below.</div>", unsafe_allow_html=True)
    else:
        if not all(c in df.columns for c in REQUIRED_COLS):
            st.error(f"Uploaded CSV missing required columns: {REQUIRED_COLS}")
        else:
            X = df[REQUIRED_COLS].copy()
            y = df['cpu_usage'].astype(float) if 'cpu_usage' in df.columns else None

            # local predictions
            if model is not None:
                try:
                    preds_local = model.predict(X)
                    df['pred_local_best'] = preds_local
                except Exception:
                    st.warning("Local model prediction failed on dataset.")

            # option to include selected MLflow runs
            runs_df = fetch_mlflow_runs()
            runs_available = []
            if isinstance(runs_df, pd.DataFrame) and not runs_df.empty:
                for _, r in runs_df.iterrows():
                    model_name = r.get('model') or 'model'
                    rmse_val = r.get('rmse')
                    label = f"{model_name} (rmse={rmse_val:.4f}) [{r.run_id[:8]}]" if rmse_val is not None else f"{model_name} [{r.run_id[:8]}]"
                    runs_available.append((label, r.run_id, model_name))
                labels = [t[0] for t in runs_available]
                default_sel = labels[:5] if len(labels) >= 5 else labels
                selected_labels = st.multiselect("Select MLflow runs to include (dataset mode)", options=labels, default=default_sel)
            else:
                selected_labels = []

            if selected_labels:
                label_to_run = {t[0]: t[1] for t in runs_available}
                loaded = 0
                failed = []
                with st.spinner("Loading selected MLflow models..."):
                    for lbl in selected_labels:
                        run_id = label_to_run.get(lbl)
                        if run_id is None:
                            continue
                        mdl = try_load_mlflow_model(run_id)
                        col_base = lbl.split()[0]  # model name prefix
                        col_name = f"pred_{col_base}_{run_id[:6]}"
                        if mdl is None:
                            failed.append(lbl)
                            continue
                        try:
                            df[col_name] = mdl.predict(X)
                            loaded += 1
                        except Exception:
                            failed.append(lbl)
                st.markdown(f"<div class='small tight'>Loaded predictions from <b>{loaded}</b> models.</div>", unsafe_allow_html=True)
                if failed:
                    st.markdown(f"<div class='small tight'>Failed: {', '.join(failed[:6])}{'...' if len(failed)>6 else ''}</div>", unsafe_allow_html=True)

            # show sample table and metrics
            pred_cols = [c for c in df.columns if c.startswith('pred_') or c == 'pred_local_best']
            display_cols = ['cpu_request', 'cpu_limit', 'runtime_minutes', 'controller_kind', 'mem_request', 'mem_limit']
            if 'cpu_usage' in df.columns:
                display_cols.append('cpu_usage')
            display_cols += pred_cols
            st.markdown("<div style='margin-top:6px'><b>Sample predictions (first 200 rows)</b></div>", unsafe_allow_html=True)
            st.dataframe(df[display_cols].head(200), use_container_width=True, height=340)

            # metrics table if actual present
            if 'cpu_usage' in df.columns and pred_cols:
                stats = []
                for c in pred_cols:
                    try:
                        rm = mean_squared_error(df['cpu_usage'], df[c], squared=False)
                        r2 = r2_score(df['cpu_usage'], df[c])
                        stats.append({'model_col': c, 'rmse': rm, 'r2': r2})
                    except Exception:
                        stats.append({'model_col': c, 'rmse': None, 'r2': None})
                stats_df = pd.DataFrame(stats).sort_values('rmse')
                st.markdown("<div style='margin-top:6px'><b>Metrics (on uploaded data)</b></div>", unsafe_allow_html=True)
                st.dataframe(stats_df.reset_index(drop=True), use_container_width=True, height=220)

            # download combined predictions
            csv_buf = io.StringIO()
            df.to_csv(csv_buf, index=False)
            st.download_button("Download all predictions CSV", csv_buf.getvalue(), file_name="predictions_all.csv", key="dl_all")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")  # spacer

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<b>Artifacts & Feature importances</b>", unsafe_allow_html=True)
    if os.path.exists(ARTIFACT_FI):
        fi = read_csv_safe(ARTIFACT_FI)
        if fi is not None and not fi.empty:
            fi = fi.sort_values("importance", ascending=True)
            fig_fi = px.bar(fi, x="importance", y="feature", orientation="h", height=220)
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info("Feature importances unreadable or empty.")
    else:
        st.info("No feature importances found. Run training first.")
    if os.path.exists(ARTIFACT_RUN_SUMMARY):
        rs = read_csv_safe(ARTIFACT_RUN_SUMMARY)
        if rs is not None:
            st.markdown("<div class='small tight'><b>Local run summary</b></div>", unsafe_allow_html=True)
            st.dataframe(rs.sort_values("rmse").reset_index(drop=True), use_container_width=True, height=200)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<b>MLflow runs & timings</b>", unsafe_allow_html=True)
    runs_df = fetch_mlflow_runs()
    if (runs_df is None) or (isinstance(runs_df, pd.DataFrame) and runs_df.empty):
        runs_df = read_csv_safe(ARTIFACT_RUN_SUMMARY) or pd.DataFrame()
    if isinstance(runs_df, pd.DataFrame) and not runs_df.empty:
        display_df = runs_df.sort_values("rmse").reset_index(drop=True)
        st.dataframe(display_df, use_container_width=True, height=260)
        if "train_time_s" in display_df.columns and display_df["train_time_s"].notnull().any():
            fig_time = px.bar(display_df.sort_values("train_time_s"), x="model", y="train_time_s",
                              labels={"train_time_s": "seconds"}, height=220)
            st.plotly_chart(fig_time, use_container_width=True)
    else:
        st.info("No MLflow runs found. Run training script to create runs.")
    st.markdown("</div>", unsafe_allow_html=True)

# -------- SINGLE-ROW MODEL-COMPARISON TABLE ----------
st.markdown("")  # spacer
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("<b>Single-row model comparison</b>", unsafe_allow_html=True)

if predict_button:
    if model is None:
        st.error("Local model not loaded. Load local model or run training first.")
    else:
        single_input = pd.DataFrame([{
            "cpu_request": cpu_request,
            "mem_request": mem_request,
            "cpu_limit": cpu_limit,
            "mem_limit": mem_limit,
            "runtime_minutes": runtime_minutes,
            "controller_kind": controller_kind
        }])

        results = []

        # local model prediction
        try:
            pred_local = model.predict(single_input)[0]
            results.append({"model": "local_best_model", "prediction": float(pred_local)})
        except Exception as e:
            st.warning(f"Local model prediction error: {e}")

        # MLflow models predictions
        runs_df = fetch_mlflow_runs()
        if isinstance(runs_df, pd.DataFrame) and not runs_df.empty:
            loaded = 0
            failed = []
            with st.spinner("Loading MLflow models and predicting..."):
                for _, row in runs_df.iterrows():
                    run_id = row['run_id']
                    model_name = row.get('model') or f"run_{run_id[:6]}"
                    m = try_load_mlflow_model(run_id)
                    if m is None:
                        failed.append(f"{model_name} ({run_id[:6]})")
                        continue
                    try:
                        p = m.predict(single_input)[0]
                        results.append({"model": model_name, "prediction": float(p)})
                        loaded += 1
                    except Exception:
                        failed.append(f"{model_name} ({run_id[:6]})")
            if loaded == 0:
                st.info("No MLflow models could be loaded for prediction (or none available).")
            if failed:
                st.markdown(f"<div class='small tight'>Failed to load/predict: {', '.join(failed[:6])}{'...' if len(failed)>6 else ''}</div>", unsafe_allow_html=True)

        # show results table
        if results:
            table_df = pd.DataFrame(results).sort_values("prediction").reset_index(drop=True)
            table_df['prediction'] = table_df['prediction'].map(lambda x: round(x, 6))
            st.markdown("<div class='small tight'><b>Model-wise Predictions (sorted)</b></div>", unsafe_allow_html=True)
            st.table(table_df)
        else:
            st.info("No predictions available from any model.")
else:
    st.markdown("<div class='small tight'>Fill single-row inputs above and click 'Predict single' to see model-wise predictions.</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# -------- FOOTER ----------
st.markdown("<div class='small tight'>Built for model comparison • Loads selected MLflow runs and shows predictions side-by-side • Artifacts in <code>artifacts/</code></div>", unsafe_allow_html=True)
