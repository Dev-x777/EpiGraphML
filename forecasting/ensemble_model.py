"""
forecasting/ensemble_model.py — XGBoost case count forecaster

Uses XGBoost's native DMatrix API (no scikit-learn dependency).
Combines:
  - 7-day lag features  (rolling window)
  - Demographics        (population, density_per_sq_km)
  - GNN embeddings      (16-dim GraphSAGE vectors)
"""

import os
import json

import numpy as np
import pandas as pd
import xgboost as xgb


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


def train_ensemble():
    print("Training XGBoost ensemble for case count forecasting...")
    os.makedirs("forecasting", exist_ok=True)

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # ── Load data ────────────────────────────────────────────
    df_demo  = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "processed", "tn_demographics.csv"))
    df_cases = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "processed", "tn_cases.csv"))

    with open(os.path.join(PROJECT_ROOT, "data", "processed", "gnn_embeddings.json")) as f:
        embeddings = json.load(f)

    # Deduplicate region demographics
    feature_cols = ["population", "density_per_sq_km"]
    df_demo_agg  = df_demo.groupby("region_id")[feature_cols].mean().reset_index()

    # ── Build dataset with 7-day rolling window ───────────────
    X_data, y_data = [], []
    window = 7

    for r_id in df_demo_agg["region_id"]:
        r_cases = (
            df_cases[df_cases["region_id"] == r_id]
            .sort_values("date")["cases"]
            .values
        )
        r_demo = df_demo_agg[df_demo_agg["region_id"] == r_id].iloc[0]
        r_emb  = embeddings.get(r_id, [0.0] * 16)

        for i in range(len(r_cases) - window):
            lag_features   = r_cases[i : i + window].tolist()
            target         = r_cases[i + window]
            feature_vector = (
                lag_features
                + [r_demo["population"], r_demo["density_per_sq_km"]]
                + r_emb
            )
            X_data.append(feature_vector)
            y_data.append(target)

    X = np.array(X_data, dtype=np.float32)
    y = np.array(y_data, dtype=np.float32)

    print(f"Dataset shape: {X.shape}")

    # ── Train / test split (80 / 20, shuffled) ───────────────
    np.random.seed(42)
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    split = int(0.8 * len(X))

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest  = xgb.DMatrix(X_test,  label=y_test)

    # ── Train with native API ────────────────────────────────
    params = {
        "objective":        "reg:squarederror",
        "max_depth":        5,
        "learning_rate":    0.1,
        "n_estimators":     100,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "seed":             42,
        "verbosity":        0,
    }

    evals_result = {}
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=[(dtest, "test")],
        evals_result=evals_result,
        verbose_eval=False,
    )

    # ── Evaluate ─────────────────────────────────────────────
    preds = booster.predict(dtest)
    rmse_val = np.sqrt(mse(y_test, preds))
    mae_val  = mae(y_test, preds)
    r2_val   = r2(y_test, preds)

    print("\nModel Evaluation (XGBoost + GNN embeddings):")
    print(f"  RMSE : {rmse_val:.2f}")
    print(f"  MAE  : {mae_val:.2f}")
    print(f"  R2   : {r2_val:.4f}")

    # ── Save booster ──────────────────────────────────────────
    model_path = os.path.join(PROJECT_ROOT, "forecasting", "xgb_model.json")
    booster.save_model(model_path)
    print(f"\nModel saved -> {model_path}")

    return rmse_val, mae_val, r2_val


if __name__ == "__main__":
    train_ensemble()
