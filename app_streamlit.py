# app_streamlit_full.py
import os
import io
import time
import joblib
import requests
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import requests
from pathlib import Path
import shutil
import time

def download_file(url: str, dest: Path, force: bool=False, timeout=30, retries=6, backoff=1.5):
    """
    Robust downloader:
    - Downloads to a temporary file first.
    - Verifies content length if available.
    - Retries on failure.
    - Only replaces the real file if the download is complete.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # if file exists and not forcing redownload
    if dest.exists() and not force:
        # treat empty files as invalid
        if dest.stat().st_size > 0:
            return dest

    attempt = 0
    last_exc = None

    while attempt < retries:
        attempt += 1
        try:
            resp = requests.get(url, stream=True, timeout=timeout)
            resp.raise_for_status()

            expected_size = resp.headers.get("Content-Length")
            if expected_size is not None:
                expected_size = int(expected_size)

            tmp_path = dest.with_suffix(".tmp")
            with open(tmp_path, "wb") as f:
                bytes_written = 0
                for chunk in resp.iter_content(8192):
                    if chunk:
                        f.write(chunk)
                        bytes_written += len(chunk)

            # verify integrity
            if expected_size is not None and bytes_written != expected_size:
                tmp_path.unlink(missing_ok=True)
                raise IOError(f"Incomplete download: expected {expected_size}, got {bytes_written}")

            # atomic move
            shutil.move(str(tmp_path), str(dest))
            return dest

        except Exception as e:
            last_exc = e
            time.sleep(backoff * attempt)
            continue

    raise RuntimeError(f"Failed to download {url} after {retries} attempts: {last_exc}")

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Cat CPU Predictor — Good vs Villain", layout="wide", initial_sidebar_state="expanded")

# Remote resources (from user)
RF_URL = "https://modelrf.blob.core.windows.net/anything/models/rf.pkl"
GB_URL = "https://modelrf.blob.core.windows.net/anything/models/gb.pkl"
RUN_SUM_URL = "https://modelrf.blob.core.windows.net/anything/run_summary.csv"

# Local cache dir for downloaded resources)
CACHE_DIR = Path(".cached_models")
CACHE_DIR.mkdir(exist_ok=True)

RF_LOCAL = CACHE_DIR / "rf.pkl"
GB_LOCAL = CACHE_DIR / "gb.pkl"
RUN_SUM_LOCAL = CACHE_DIR / "run_summary.csv"

# attempt to download RF and GB
try:
    download_file(RF_URL, RF_LOCAL)
except Exception as e:
    st.error(f"RF download failed: {e}")

GB_LOCAL = CACHE_DIR / "gb.pkl"
RUN_SUM_LOCAL = CACHE_DIR / "run_summary.csv"

REQUIRED_COLS = ['cpu_request','mem_request','cpu_limit','mem_limit','runtime_minutes','controller_kind']

# Default colors
DEFAULT_PINK = "#ff66b3"   # RandomForest (hero / pink)
DEFAULT_RED = "#ff1a1a"    # GradientBoosting (villain / red)
DEFAULT_BG = "#fff6fb"

# Cat GIF (replace if you want)
CAT_GIF = "https://media.giphy.com/media/JIX9t2j0ZTN9S/giphy.gif"

# ---------------- UTIL FUNCTIONS ----------------
def download_file(url: str, dest: Path, force: bool=False, timeout=30):
    """
    Download a file from URL to dest. If dest exists and force is False, do nothing.
    Raises exceptions on network errors.
    """
    if dest.exists() and not force:
        return dest
    resp = requests.get(url, stream=True, timeout=timeout)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return dest

@st.cache_resource
def load_model_from_file(path: Path):
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Failed to load model {path.name}: {e}")
        return None

@st.cache_data(ttl=60*60)  # cache run summary for 1 hour
def load_run_summary(local_path: Path, url: str):
    # prefer local cached file; attempt download otherwise
    try:
        download_file(url, local_path)
        df = pd.read_csv(local_path)
        return df
    except Exception:
        # fallback: try reading directly from URL
        try:
            return pd.read_csv(url)
        except Exception as e:
            st.warning(f"Could not load run summary: {e}")
            return pd.DataFrame()

def bytes_to_mb(x):
    try:
        return float(x) / (1024**2)
    except Exception:
        return np.nan

def blend_hex(c1, c2, t):
    """Blend hex colors c1 -> c2 by t in [0,1]"""
    def h2rgb(h):
        h = h.lstrip("#")
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    def rgb2hex(rgb):
        return "#{:02x}{:02x}{:02x}".format(*[int(max(0,min(255,round(v)))) for v in rgb])
    r1,g1,b1 = h2rgb(c1)
    r2,g2,b2 = h2rgb(c2)
    r = r1*(1-t)+r2*t
    g = g1*(1-t)+g2*t
    b = b1*(1-t)+b2*t
    return rgb2hex((r,g,b))

# ---------------- THEME / UI CONTROLS ----------------
st.sidebar.markdown("### Theme controls")
blend_slider = st.sidebar.slider("Blend Pink ↔ Red (0=pink, 100=red)", 0, 100, 30)
t = blend_slider / 100.0
COLOR_PINK = blend_hex(DEFAULT_PINK, DEFAULT_RED, 1 - t)  # lean pink when slider low
COLOR_RED = blend_hex(DEFAULT_RED, DEFAULT_PINK, t)      # lean red when slider high
BG_LEFT = blend_hex("#fff0f6", "#ffe6e6", t)
BG_RIGHT = blend_hex("#fff6fb", "#fff0f6", t)

# Inject dynamic CSS
dynamic_css = f"""
<style>
.stApp {{
    background: linear-gradient(180deg, {BG_LEFT} 0%, {BG_RIGHT} 100%);
    color: #3b0b2f;
}}
.card {{
    background: #ffffff;
    border-radius: 12px;
    padding: 14px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.06);
    border: 1px solid rgba(0,0,0,0.04);
    margin-bottom: 10px;
}}
.big-title {{ font-size: 26px; font-weight: 700; color: {blend_hex('#7d0633','#7d0633', t)}; margin:0; }}
.small {{ font-size:13px; color:#6b2136; }}
.badge-pink {{ background:{COLOR_PINK}; color:white; padding:6px 8px; border-radius:8px; font-weight:600; }}
.badge-red {{ background:{COLOR_RED}; color:white; padding:6px 8px; border-radius:8px; font-weight:600; }}
footer {{ visibility: hidden; }}
</style>
"""
st.markdown(dynamic_css, unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<div class='card' style='display:flex;align-items:center;justify-content:space-between'>", unsafe_allow_html=True)
st.markdown(f"<div><div class='big-title'>Cat CPU Predictor 🐱</div><div class='small'>Good (pink) vs Villain (red) — RandomForest vs GradientBoosting</div></div>", unsafe_allow_html=True)
st.image(CAT_GIF, width=110)
st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Download & Load Models ----------------
col_r, col_g, col_ctrl = st.columns([1,1,1], gap="small")
with col_ctrl:
    if st.button("Refresh cached models & run summary"):
        try:
            download_file(RF_URL, RF_LOCAL, force=True)
            download_file(GB_URL, GB_LOCAL, force=True)
            download_file(RUN_SUM_URL, RUN_SUM_LOCAL, force=True)
            st.success("Refreshed cached files. Rerunning app...")
            time.sleep(0.6)
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Refresh failed: {e}")

# Attempt download (no force) - errors are shown but app continues
try:
    download_file(RF_URL, RF_LOCAL)
except Exception as e:
    st.warning(f"Could not download RF model: {e}")
try:
    download_file(GB_URL, GB_LOCAL)
except Exception as e:
    st.warning(f"Could not download GB model: {e}")
try:
    download_file(RUN_SUM_URL, RUN_SUM_LOCAL)
except Exception:
    pass  # load_run_summary will try URL if local failed

rf_model = load_model_from_file(RF_LOCAL)
gb_model = load_model_from_file(GB_LOCAL)

# quick status cards
col1, col2, col3 = st.columns([1,1,2], gap="small")
with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-weight:700;color:{COLOR_PINK}'>💖 RandomForest (Hero)</div>")
    st.markdown(f"- file: `{RF_LOCAL.name}`")
    st.markdown(f"- status: **{'loaded' if rf_model is not None else 'missing'}**")
    st.markdown("</div>", unsafe_allow_html=True)
with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-weight:700;color:{COLOR_RED}'>🔴 GradientBoosting (Villain)</div>")
    st.markdown(f"- file: `{GB_LOCAL.name}`")
    st.markdown(f"- status: **{'loaded' if gb_model is not None else 'missing'}**")
    st.markdown("</div>", unsafe_allow_html=True)
with col3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    run_df = load_run_summary(RUN_SUM_LOCAL, RUN_SUM_URL)
    st.markdown("📊 Run summary")
    if run_df.empty:
        st.markdown("<div class='small'>No run summary found (cached or remote).</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='small'>Loaded {len(run_df)} runs (from run_summary.csv)</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# ---------------- RUN SUMMARY VISUALIZATION (robust block) ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### Model comparison (run summary)")

if run_df.empty:
    st.info("Run summary is empty or missing.")
else:
    # normalize numeric columns
    for c in ['rmse', 'r2', 'train_time_seconds']:
        if c in run_df.columns:
            run_df[c] = pd.to_numeric(run_df[c], errors='coerce')

    # normalize model column: prefer 'model' then 'name' then alternatives
    if 'model' not in run_df.columns:
        if 'name' in run_df.columns:
            run_df = run_df.rename(columns={'name': 'model'})
        else:
            for alt in ['Model', 'model_name', 'run_name']:
                if alt in run_df.columns:
                    run_df['model'] = run_df[alt].astype(str)
                    break

    # fallback synthetic names
    if 'model' not in run_df.columns:
        if 'run_id' in run_df.columns:
            run_df['model'] = run_df['run_id'].astype(str)
        else:
            run_df = run_df.reset_index(drop=True)
            run_df['model'] = run_df.index.map(lambda i: f"run_{i}")

    # color mapping function
    def model_color(name):
        s = str(name).lower()
        if "randomforest" in s or s.startswith("rf") or "rf" in s:
            return COLOR_PINK
        if "gradientboost" in s or s.startswith("gb") or "gb" in s:
            return COLOR_RED
        return "#888888"

    run_df['color'] = run_df['model'].apply(lambda x: model_color(x))

    # controls
    models = sorted(run_df['model'].astype(str).unique())
    sel_models = st.multiselect("Select models", options=models, default=models)

    df_filtered = run_df[run_df['model'].astype(str).isin(sel_models)].copy()
    if df_filtered.empty:
        st.warning("No models selected or no matching rows.")
    else:
        st.dataframe(df_filtered.reset_index(drop=True), use_container_width=True, height=220)

        c1, c2 = st.columns(2, gap="small")
        if "rmse" in df_filtered.columns:
            fig_rmse = px.bar(
                df_filtered.sort_values("rmse"),
                x="model", y="rmse",
                color="model",
                color_discrete_map={m: model_color(m) for m in df_filtered['model'].unique()},
                title="RMSE by Model"
            )
            fig_rmse.update_layout(xaxis_tickangle=-45, showlegend=False, margin=dict(t=40,b=120))
            c1.plotly_chart(fig_rmse, use_container_width=True)
        if "r2" in df_filtered.columns:
            fig_r2 = px.bar(
                df_filtered.sort_values("r2", ascending=False),
                x="model", y="r2",
                color="model",
                color_discrete_map={m: model_color(m) for m in df_filtered['model'].unique()},
                title="R² by Model"
            )
            fig_r2.update_layout(xaxis_tickangle=-45, showlegend=False, margin=dict(t=40,b=120))
            c2.plotly_chart(fig_r2, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- UPLOAD CSV / BATCH PREDICTIONS ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("## Upload CSV for batch predictions")
uploaded = st.file_uploader("Upload CSV (must include required columns)", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.dataframe(df.head(6), use_container_width=True)
    if not all(c in df.columns for c in REQUIRED_COLS):
        st.error(f"CSV missing required columns: {REQUIRED_COLS}")
    else:
        X = df[REQUIRED_COLS].copy()
        # convert mem bytes to MB as training pipeline expects
        X['mem_request'] = X['mem_request'].apply(bytes_to_mb)
        X['mem_limit'] = X['mem_limit'].apply(bytes_to_mb)

        if rf_model is not None:
            try:
                df['pred_rf'] = rf_model.predict(X)
            except Exception as e:
                st.error(f"RF predict failed: {e}")
        if gb_model is not None:
            try:
                df['pred_gb'] = gb_model.predict(X)
            except Exception as e:
                st.error(f"GB predict failed: {e}")

        st.markdown("### Predictions sample")
        st.dataframe(df.head(20), use_container_width=True)

        # download
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        st.download_button("Download predictions CSV", buf.getvalue(), file_name="predictions.csv")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# ---------------- SINGLE-ROW PREDICTION ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("## Single-row prediction — Good (pink) vs Villain (red)")

left, right = st.columns(2, gap="small")
with left:
    cpu_request = st.number_input("cpu_request", value=0.2, step=0.01, format="%.3f")
    cpu_limit = st.number_input("cpu_limit", value=0.5, step=0.01, format="%.3f")
    runtime_minutes = st.number_input("runtime_minutes", value=0.0, format="%.1f")
with right:
    mem_request = st.number_input("mem_request (bytes)", value=134217728, format="%d")
    mem_limit = st.number_input("mem_limit (bytes)", value=268435456, format="%d")
    controller_kind = st.selectbox("controller_kind", options=["Job","ReplicaSet","Deployment","Unknown"])

if st.button("Predict (Good vs Villain)"):
    single = {
        "cpu_request": cpu_request,
        "mem_request": bytes_to_mb(mem_request),
        "cpu_limit": cpu_limit,
        "mem_limit": bytes_to_mb(mem_limit),
        "runtime_minutes": runtime_minutes,
        "controller_kind": controller_kind
    }
    input_df = pd.DataFrame([single])

    results = []
    if rf_model is not None:
        try:
            p = rf_model.predict(input_df)[0]
            results.append(("RandomForest", p))
        except Exception as e:
            st.error(f"RF predict error: {e}")
    if gb_model is not None:
        try:
            p = gb_model.predict(input_df)[0]
            results.append(("GradientBoosting", p))
        except Exception as e:
            st.error(f"GB predict error: {e}")

    if results:
        rows_html = ""
        for name, pred in results:
            color = COLOR_PINK if "RandomForest" in name else COLOR_RED
            badge = "💖 Hero" if "RandomForest" in name else "🔴 Villain"
            rows_html += f"""
            <tr>
                <td style="padding:8px;border-bottom:1px solid #eee"><b>{name}</b></td>
                <td style="padding:8px;border-bottom:1px solid #eee">{pred:.6f}</td>
                <td style="padding:8px;border-bottom:1px solid #eee"><span style="background:{color};color:white;padding:6px;border-radius:8px">{badge}</span></td>
            </tr>
            """

        table_html = f"""
        <table style="width:100%; border-collapse:collapse; font-family:inherit">
            <thead>
                <tr><th style="text-align:left;padding:8px">Model</th><th style="text-align:left;padding:8px">Prediction</th><th style="text-align:left;padding:8px">Role</th></tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
        """
        st.markdown(table_html, unsafe_allow_html=True)
    else:
        st.info("No models available for prediction. Ensure models downloaded successfully.")
st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown(f"<div style='margin-top:14px;text-align:center;color:{blend_hex('#7d0633','#7d0633', t)}'>Made with ❤️ — Cat theme, Good (pink) vs Villain (red). Models cached in <code>{CACHE_DIR}</code></div>", unsafe_allow_html=True)
