# app_streamlit_cat_cyberpunk.py
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
import shutil

# ----------------- CONFIG -----------------
st.set_page_config(
    page_title="CPU Predictor — Cyberpunk",
    layout="wide",
    initial_sidebar_state="expanded",
)

# remote resources (use your blob links)
RF_URL = "https://modelrf.blob.core.windows.net/anything/models/rf.pkl"
GB_URL = "https://modelrf.blob.core.windows.net/anything/models/gb.pkl"
RUN_SUM_URL = "https://modelrf.blob.core.windows.net/anything/run_summary.csv"

# local cache dir
CACHE_DIR = Path(".cached_models")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

RF_LOCAL = CACHE_DIR / "rf.pkl"
GB_LOCAL = CACHE_DIR / "gb.pkl"
RUN_SUM_LOCAL = CACHE_DIR / "run_summary.csv"

REQUIRED_COLS = [
    "cpu_request",
    "mem_request",
    "cpu_limit",
    "mem_limit",
    "runtime_minutes",
    "controller_kind",
]

# cyberpunk palette & fonts
NEON_CYAN = "#00f6ff"
NEON_MAGENTA = "#ff00d6"
DEEP_BG = "#061226"
GLASS = "rgba(255,255,255,0.03)"
TEXT_NEON = "#cbe9ff"
MUTED = "#7ea6b9"
ACCENT = NEON_MAGENTA

# minimal cyberpunk CSS
# uses google font 'Orbitron' for futuristic feel
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');

    :root {{
        --bg: {DEEP_BG};
        --glass: {GLASS};
        --neon-cyan: {NEON_CYAN};
        --neon-magenta: {NEON_MAGENTA};
        --text-neon: {TEXT_NEON};
        --muted: {MUTED};
    }}

    html, body, .stApp {{
        background: radial-gradient(1200px 600px at 10% 10%, rgba(0,246,255,0.06), transparent 10%),
                    radial-gradient(800px 400px at 90% 90%, rgba(255,0,214,0.06), transparent 12%),
                    var(--bg);
        font-family: 'Orbitron', sans-serif;
        color: var(--text-neon);
    }}

    /* card-like containers */
    .card {{
        background: linear-gradient(180deg, rgba(255,255,255,0.02) 0%, rgba(255,255,255,0.015) 100%);
        border: 1px solid rgba(255,255,255,0.04);
        border-left: 3px solid var(--neon-cyan);
        padding: 14px;
        border-radius: 10px;
        box-shadow: 0 8px 40px rgba(0,0,0,0.6);
    }}

    .title {{
        color: var(--neon-cyan);
        font-weight:700;
        font-size:26px;
        margin:0;
    }}

    .muted {{
        color: var(--muted);
        font-size:13px;
        margin:0;
    }}

    .small {{
        font-size:13px;
        color:var(--muted);
    }}

    .badge-neon {{
        background: linear-gradient(90deg,var(--neon-cyan), var(--neon-magenta));
        color: #021017;
        padding:6px 10px;
        border-radius:8px;
        font-weight:700;
        box-shadow: 0 6px 20px rgba(0,246,255,0.06);
    }}

    /* hide default footer and hamburger tweak */
    footer {{ visibility: hidden; }}
    .css-1v0mbdj {{}}
    /* input & button neon focus */
    .stButton>button {{
        border-radius: 8px;
        background: linear-gradient(90deg,var(--neon-cyan), var(--neon-magenta));
        color: #021017;
        font-weight:700;
        box-shadow: 0 6px 18px rgba(0,0,0,0.6);
    }}

    .stNumberInput>div, .stSelectbox>div {{
        border: 1px solid rgba(255,255,255,0.04);
        border-radius:8px;
        padding:6px;
        background: rgba(255,255,255,0.01);
    }}

    /* table header neon tint */
    .stDataFrame thead th {{
        background: linear-gradient(90deg, rgba(0,246,255,0.06), rgba(255,0,214,0.04));
        color: var(--text-neon);
    }}

    /* smaller screens tweaks */
    @media (max-width: 600px) {{
        .title {{ font-size:20px; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------- UTIL: robust downloader -----------------
def robust_download(url: str, dest: Path, force: bool = False, timeout: int = 60, retries: int = 3):
    """
    Download to a temp file, verify size via Content-Length (if provided),
    retry on failure, and atomically move into place.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        if dest.stat().st_size > 0:
            return dest

    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, stream=True, timeout=timeout)
            resp.raise_for_status()
            content_length = resp.headers.get("Content-Length")
            tmp = dest.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                written = 0
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        written += len(chunk)
            # validate if header present
            if content_length is not None:
                expected = int(content_length)
                if written != expected:
                    tmp.unlink(missing_ok=True)
                    raise IOError(f"Partial download: expected {expected} bytes, got {written}")
            if written == 0:
                tmp.unlink(missing_ok=True)
                raise IOError("Downloaded file is empty (0 bytes).")
            shutil.move(str(tmp), str(dest))  # atomic
            return dest
        except Exception as e:
            last_exc = e
            time.sleep(1.2 * attempt)
            continue
    raise RuntimeError(f"Failed to download {url} after {retries} attempts. Last error: {last_exc}")

# ----------------- Model loader & run summary -----------------
@st.cache_resource
def load_model(path: Path):
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        # joblib may throw binary/pickle errors if file corrupted
        return RuntimeError(str(e))

@st.cache_data(ttl=60 * 60)
def load_run_summary(local_path: Path, url: str):
    # try local cached CSV, otherwise download
    try:
        if not local_path.exists():
            robust_download(url, local_path)
        df = pd.read_csv(local_path)
        return df
    except Exception:
        try:
            # fallback to direct read from URL (if allowed)
            return pd.read_csv(url)
        except Exception:
            return pd.DataFrame()

# ----------------- Header -----------------
col_head_left, col_head_right = st.columns([3, 1], gap="small")
with col_head_left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='title'>CPU Predictor</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='muted'>Neon, dark dashboard for CPU usage prediction — RandomForestvs GradientBoosting</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col_head_right:
    # cyberpunk cat (if you prefer a different image, replace URL)
    st.image("https://media.giphy.com/media/v1.Y2lkPWVjZjA1ZTQ3Zzhzam5idXYxaW5iam92a2Zzdm9kd3cxbnJ2d2EyN2JyYXVocDE4eSZlcD12MV9naWZzX3NlYXJjaCZjdD1n/3oEjHERaTIdeuFQrXq/giphy.gif", width=110)

st.markdown("")  # tiny spacer

# ---------- Controls: refresh, neon intensity ----------
with st.expander("Controls"):
    col_a, col_b = st.columns([1, 2], gap="small")
    with col_a:
        refresh = st.button("Refresh models & run summary")
        clear_cache = st.button("Clear local cache")
    with col_b:
        neon_bias = st.slider("Neon intensity (visual only)", min_value=0, max_value=100, value=70)

# handle cache clearing / refresh
if clear_cache:
    try:
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        st.success("Cleared local cache. Click Refresh to redownload models.")
        st.experimental_rerun()
    except Exception as e:
        st.error(f"Failed to clear cache: {e}")

# download models (only models from blob)
download_errors = []
try:
    robust_download(RF_URL, RF_LOCAL, force=refresh)
except Exception as e:
    download_errors.append(f"RF: {e}")
try:
    robust_download(GB_URL, GB_LOCAL, force=refresh)
except Exception as e:
    download_errors.append(f"GB: {e}")

# load
rf_model = load_model(RF_LOCAL)
gb_model = load_model(GB_LOCAL)

# show status card
col1, col2, col3 = st.columns([1, 1, 2], gap="small")
with col1:
    st.markdown("**RandomForest (hero)**")
    if isinstance(rf_model, RuntimeError):
        st.error(f"Failed to load RF model: {rf_model}")
    elif rf_model is None:
        st.warning("RF model not present locally.")
    else:
        st.success("RF loaded")
with col2:
    st.markdown("**GradientBoosting (villain)**")
    if isinstance(gb_model, RuntimeError):
        st.error(f"Failed to load GB model: {gb_model}")
    elif gb_model is None:
        st.warning("GB model not present locally.")
    else:
        st.success("GB loaded")
with col3:
    if download_errors:
        for err in download_errors:
            st.warning(err)
    st.markdown("**Artifacts**")
    st.markdown(f"- Run summary URL: `{RUN_SUM_URL}`")

st.markdown("---")

# ----------------- Run summary & charts -----------------
run_df = load_run_summary(RUN_SUM_LOCAL, RUN_SUM_URL)
if run_df.empty:
    st.info("No run summary found. Run training to produce run_summary.csv or check RUN_SUM_URL.")
else:
    # normalize numeric columns
    for c in ["rmse", "r2", "train_time_seconds"]:
        if c in run_df.columns:
            run_df[c] = pd.to_numeric(run_df[c], errors="coerce")

    # normalize model column (support 'name' header)
    if "model" not in run_df.columns:
        if "name" in run_df.columns:
            run_df = run_df.rename(columns={"name": "model"})
        else:
            for alt in ["Model", "model_name", "run_name"]:
                if alt in run_df.columns:
                    run_df["model"] = run_df[alt].astype(str)
                    break
    if "model" not in run_df.columns:
        if "run_id" in run_df.columns:
            run_df["model"] = run_df["run_id"].astype(str)
        else:
            run_df = run_df.reset_index(drop=True)
            run_df["model"] = run_df.index.map(lambda i: f"run_{i}")

    # map model -> color (cyan hero vs magenta villain)
    def pick_color(model_name):
        s = str(model_name).lower()
        if "randomforest" in s or "rf" in s:
            return NEON_CYAN
        if "gradientboost" in s or "gb" in s:
            return NEON_MAGENTA
        return "#888888"

    run_df["color"] = run_df["model"].apply(pick_color)

    st.markdown("### Model runs")
    sel = st.multiselect(
        "Choose models to view",
        options=sorted(run_df["model"].unique()),
        default=sorted(run_df["model"].unique()),
    )
    df_view = run_df[run_df["model"].isin(sel)].copy()
    st.dataframe(df_view.reset_index(drop=True), use_container_width=True, height=220)

    # charts
    c1, c2 = st.columns(2, gap="small")
    if "rmse" in df_view.columns:
        fig_rmse = px.bar(
            df_view.sort_values("rmse"),
            x="model",
            y="rmse",
            color="model",
            color_discrete_map={m: pick_color(m) for m in df_view["model"].unique()},
        )
        fig_rmse.update_layout(
            showlegend=False,
            xaxis_tickangle=-45,
            title="RMSE by model",
            margin=dict(b=120),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT_NEON if 'TEXT_NEON' in globals() else "#cbe9ff"),
        )
        c1.plotly_chart(fig_rmse, use_container_width=True)
    if "r2" in df_view.columns:
        fig_r2 = px.bar(
            df_view.sort_values("r2", ascending=False),
            x="model",
            y="r2",
            color="model",
            color_discrete_map={m: pick_color(m) for m in df_view["model"].unique()},
        )
        fig_r2.update_layout(
            showlegend=False,
            xaxis_tickangle=-45,
            title="R² by model",
            margin=dict(b=120),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT_NEON if 'TEXT_NEON' in globals() else "#cbe9ff"),
        )
        c2.plotly_chart(fig_r2, use_container_width=True)

st.markdown("---")

# ----------------- Upload CSV and batch predictions -----------------
st.markdown("## Batch predictions (upload CSV)")
uploaded = st.file_uploader("Upload CSV with required columns", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.dataframe(df.head(6), use_container_width=True)
    if not all(c in df.columns for c in REQUIRED_COLS):
        st.error(f"CSV missing required columns: {REQUIRED_COLS}")
    else:
        X = df[REQUIRED_COLS].copy()
        X["mem_request"] = X["mem_request"].apply(
            lambda x: float(x) / (1024 ** 2) if pd.notna(x) else x
        )
        X["mem_limit"] = X["mem_limit"].apply(
            lambda x: float(x) / (1024 ** 2) if pd.notna(x) else x
        )

        if rf_model is not None and not isinstance(rf_model, RuntimeError):
            try:
                df["pred_rf"] = rf_model.predict(X)
            except Exception as e:
                st.error(f"RF prediction failed: {e}")
        if gb_model is not None and not isinstance(gb_model, RuntimeError):
            try:
                df["pred_gb"] = gb_model.predict(X)
            except Exception as e:
                st.error(f"GB prediction failed: {e}")

        st.markdown("Predictions (sample)")
        st.dataframe(df.head(20), use_container_width=True)

        buf = io.StringIO()
        df.to_csv(buf, index=False)
        st.download_button("Download predictions CSV", buf.getvalue(), file_name="predictions.csv")

st.markdown("---")

# ----------------- Single-row prediction -----------------
st.markdown("## Single-row prediction")
left, right = st.columns(2, gap="small")
with left:
    cpu_request = st.number_input("cpu_request", value=0.2, step=0.01, format="%.3f")
    cpu_limit = st.number_input("cpu_limit", value=0.5, step=0.01, format="%.3f")
    runtime_minutes = st.number_input("runtime_minutes", value=0.0, format="%.1f")
with right:
    mem_request = st.number_input("mem_request (bytes)", value=134217728, format="%d")
    mem_limit = st.number_input("mem_limit (bytes)", value=268435456, format="%d")
    controller_kind = st.selectbox(
        "controller_kind", options=["Job", "ReplicaSet", "Deployment", "Unknown"]
    )

if st.button("Predict (RF & GB)"):
    input_row = {
        "cpu_request": cpu_request,
        "mem_request": float(mem_request) / (1024 ** 2) if mem_request is not None else np.nan,
        "cpu_limit": cpu_limit,
        "mem_limit": float(mem_limit) / (1024 ** 2) if mem_limit is not None else np.nan,
        "runtime_minutes": runtime_minutes,
        "controller_kind": controller_kind,
    }
    X_single = pd.DataFrame([input_row])

    results = []
    if rf_model is not None and not isinstance(rf_model, RuntimeError):
        try:
            p = rf_model.predict(X_single)[0]
            results.append(("RandomForest", p))
        except Exception as e:
            st.error(f"RF predict error: {e}")
    if gb_model is not None and not isinstance(gb_model, RuntimeError):
        try:
            p = gb_model.predict(X_single)[0]
            results.append(("GradientBoosting", p))
        except Exception as e:
            st.error(f"GB predict error: {e}")

    if results:
        res_df = pd.DataFrame(results, columns=["model", "prediction"]).sort_values(
            "prediction"
        ).reset_index(drop=True)
        # show with simple table; for extra neon flair we keep header styling via CSS above
        st.table(res_df)
    else:
        st.info("No models available to predict (or models failed to load).")

st.markdown("---")
st.caption(
    "Made with ❤️ — Cyberpunk cat theme. Replace the cat image URL near the header if you want a different mascot."
)
