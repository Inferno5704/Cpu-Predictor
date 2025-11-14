# train_save_models.py
import argparse
import os
import time
import joblib
import warnings
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score

warnings.filterwarnings("ignore")

NUMERIC = ['cpu_request','mem_request','cpu_limit','mem_limit','runtime_minutes']
CATEGORICAL = ['controller_kind']

def rmse(y_true, y_pred): return root_mean_squared_error(y_true, y_pred)

def load_prepare(path):
    df = pd.read_csv(path)
    # keep required columns if present
    keep = [c for c in (NUMERIC + CATEGORICAL + ['cpu_usage']) if c in df.columns]
    df = df[keep].copy()
    df = df.dropna(subset=['cpu_usage'])
    df['runtime_minutes'] = df['runtime_minutes'].fillna(0)
    for col in ['cpu_request','mem_request','cpu_limit','mem_limit']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    if 'controller_kind' in df.columns:
        df['controller_kind'] = df['controller_kind'].fillna('Unknown')
    else:
        df['controller_kind'] = 'Unknown'
    # convert bytes -> MB (same transformation used in Streamlit)
    if 'mem_request' in df.columns:
        df['mem_request'] = df['mem_request'] / (1024**2)
    if 'mem_limit' in df.columns:
        df['mem_limit'] = df['mem_limit'] / (1024**2)
    return df

def build_preproc():
    num_transform = Pipeline([('scaler', StandardScaler())])
    cat_transform = Pipeline([('ohe', OneHotEncoder(handle_unknown='ignore'))])
    preproc = ColumnTransformer([
        ('num', num_transform, NUMERIC),
        ('cat', cat_transform, CATEGORICAL)
    ], remainder='drop')
    return preproc

def main(args):
    os.makedirs(args.output, exist_ok=True)
    models_dir = os.path.join(args.output, "models")
    os.makedirs(models_dir, exist_ok=True)

    df = load_prepare(args.data)
    X = df[NUMERIC + CATEGORICAL].copy()
    y = df['cpu_usage'].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42)

    preproc = build_preproc()

    candidates = {
        "rf": RandomForestRegressor(n_estimators=args.n_estimators_rf, n_jobs=-1, random_state=42),
        "gb": GradientBoostingRegressor(n_estimators=args.n_estimators_gb, random_state=42)
    }

    summaries = []
    best_rmse = float('inf')
    best_name = None

    for key, estimator in candidates.items():
        name = "RandomForest" if key == "rf" else "GradientBoosting"
        print(f"Training {name} ...")
        start = time.time()
        pipeline = Pipeline([('preproc', preproc), ('model', estimator)])
        pipeline.fit(X_train, y_train)
        elapsed = time.time() - start

        preds = pipeline.predict(X_test)
        val_rmse = rmse(y_test, preds)
        r2 = r2_score(y_test, preds)
        summaries.append({'model': name, 'rmse': float(val_rmse), 'r2': float(r2), 'train_time_seconds': float(elapsed)})

        # save each pipeline
        out_path = os.path.join(models_dir, f"{key}.pkl")
        joblib.dump(pipeline, out_path)
        print(f"Saved {name} to {out_path} (RMSE={val_rmse:.6f}, R2={r2:.4f}, time={elapsed:.1f}s)")

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_name = (key, pipeline)

    # save best as best_model.pkl for backward compatibility
    if best_name is not None:
        joblib.dump(best_name[1], os.path.join(args.output, "best_model.pkl"))
        print(f"Best model ({best_name[0]}) saved to {os.path.join(args.output, 'best_model.pkl')}")

    # run summary
    pd.DataFrame(summaries).to_csv(os.path.join(args.output, "run_summary.csv"), index=False)
    print("Wrote run_summary.csv")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="runner.csv", help="Path to CSV")
    p.add_argument("--output", default="artifacts2", help="Output artifacts dir")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--n_estimators_rf", type=int, default=40)
    p.add_argument("--n_estimators_gb", type=int, default=200)
    args = p.parse_args()
    main(args)
