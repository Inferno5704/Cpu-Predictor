# train_and_track.py
"""
Train multiple regression models for CPU usage and track experiments with MLflow.
- Defaults: local MLflow, RMSE metric, one-hot encoding for controller_kind.
- Outputs: artifacts/best_model.pkl and artifacts/feature_importances.csv (if available)
- Logs: model, rmse, r2, train_time_seconds to MLflow; sample predictions artifact.
"""
import argparse
import os
import time
import warnings
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from tqdm import tqdm

warnings.filterwarnings("ignore")

NUMERIC = ['cpu_request','mem_request','cpu_limit','mem_limit','runtime_minutes']
CATEGORICAL = ['controller_kind']
ALL_COLS = NUMERIC + CATEGORICAL + ['cpu_usage']

def rmse(y_true, y_pred): return root_mean_squared_error(y_true, y_pred)

def load_and_prepare(path):
    df = pd.read_csv(path)
    df = df.loc[:, df.columns.intersection(ALL_COLS)].copy()
    df = df.dropna(subset=['cpu_usage'])
    # sensible imputations
    df['runtime_minutes'] = df['runtime_minutes'].fillna(0)
    for col in ['cpu_request','mem_request','cpu_limit','mem_limit']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    df['controller_kind'] = df.get('controller_kind', pd.Series(['Unknown']*len(df))).fillna('Unknown')
    return df

def build_preprocessor():
    num_transform = Pipeline([('scaler', StandardScaler())])
    cat_transform = Pipeline([('ohe', OneHotEncoder(handle_unknown='ignore'))])
    preproc = ColumnTransformer([
        ('num', num_transform, NUMERIC),
        ('cat', cat_transform, CATEGORICAL)
    ], remainder='drop')
    return preproc

def candidate_models():
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.01, max_iter=5000),
        'RandomForest': RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=100, n_jobs=-1, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'HistGradientBoosting': HistGradientBoostingRegressor(max_iter=100, random_state=42)
    }
    # optionally include xgboost & lightgbm if installed
    try:
        from xgboost import XGBRegressor
        models['XGBoost'] = XGBRegressor(n_estimators=100, use_label_encoder=False, eval_metric='rmse', verbosity=0, n_jobs=-1, random_state=42)
    except Exception:
        pass
    try:
        from lightgbm import LGBMRegressor
        models['LightGBM'] = LGBMRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    except Exception:
        pass
    return models

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    mlflow.set_experiment(args.mlflow_experiment)

    df = load_and_prepare(args.data)
    X = df[NUMERIC + CATEGORICAL].copy()
    y = df['cpu_usage'].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42)

    preproc = build_preprocessor()
    candidates = candidate_models()

    best_rmse = float('inf')
    best_pipeline = None
    best_name = None
    run_summaries = []

    print(f"Found {len(candidates)} candidate models: {list(candidates.keys())}\n")

    for name, estimator in candidates.items():
        print(f"--- Training {name} ---")
        start_time = time.time()
        with mlflow.start_run(run_name=name):
            pipeline = Pipeline([('preproc', preproc), ('model', estimator)])
            # show a terminal progress indicator (simple)
            print(f"[{name}] fitting...", end="", flush=True)
            pipeline.fit(X_train, y_train)
            elapsed = time.time() - start_time
            print(f" done. (train_time={elapsed:.1f}s)")

            preds = pipeline.predict(X_test)
            val_rmse = rmse(y_test, preds)
            r2 = r2_score(y_test, preds)
            input_example = X_train.head(3)
            signature = infer_signature(X_train, pipeline.predict(X_train))
            # Log params & metrics
            mlflow.log_param("model", name)
            mlflow.log_metric("rmse", float(val_rmse))
            mlflow.log_metric("r2", float(r2))
            mlflow.log_metric("train_time_seconds", float(elapsed))

            # Save sample predictions artifact
            sample = pd.DataFrame({'y_true': y_test.values[:100], 'y_pred': preds[:100]})
            sample_path = os.path.join(args.output_dir, f"sample_preds_{name}.csv")
            sample.to_csv(sample_path, index=False)
            mlflow.log_artifact(sample_path)
            # log model
            mlflow.sklearn.log_model(pipeline, name="model",signature=signature, input_example=input_example)
            # cleanup artifact file
            try:
                os.remove(sample_path)
            except Exception:
                pass

            print(f"{name} -> RMSE={val_rmse:.6f} R2={r2:.4f} train_time={elapsed:.1f}s")
            run_summaries.append({'name': name, 'rmse': float(val_rmse), 'r2': float(r2), 'train_time_seconds': float(elapsed)})

            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_pipeline = pipeline
                best_name = name

    # save best model locally
    if best_pipeline is not None:
        best_path = os.path.join(args.output_dir, "best_model.pkl")
        joblib.dump(best_pipeline, best_path)
        print(f"\nBest model: {best_name} with RMSE={best_rmse:.6f} saved to {best_path}")
    else:
        print("No model trained successfully.")

    # try extract feature importances for tree-based best model
    try:
        model_step = best_pipeline.named_steps['model']
        if hasattr(model_step, "feature_importances_"):
            preproc = best_pipeline.named_steps['preproc']
            # numeric names + categorical encoder names (requires fitted ohe)
            num_names = NUMERIC
            cat_ohe = preproc.named_transformers_['cat'].named_steps['ohe']
            cat_names = list(cat_ohe.get_feature_names_out(CATEGORICAL))
            feat_names = num_names + cat_names
            importances = model_step.feature_importances_
            fi_df = pd.DataFrame({'feature': feat_names, 'importance': importances})
            fi_df = fi_df.sort_values('importance', ascending=False)
            fi_path = os.path.join(args.output_dir, "feature_importances.csv")
            fi_df.to_csv(fi_path, index=False)
            print(f"Feature importances saved to {fi_path}")
    except Exception as e:
        print("Feature importances not available:", e)

    # also save run summary
    summary_df = pd.DataFrame(run_summaries)
    summary_df.to_csv(os.path.join(args.output_dir, "run_summary.csv"), index=False)
    print("Run summary written to artifacts/run_summary.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="runner.csv", help="Path to input CSV data")
    parser.add_argument("--output_dir", default="artifacts", help="Artifacts output dir")
    parser.add_argument("--mlflow_experiment", default="cpu_usage_experiment", help="MLflow experiment name")
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()
    main(args)
