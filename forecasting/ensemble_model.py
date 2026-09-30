import os
import json
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)
    
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def train_ensemble():
    print("Training XGBoost ensemble for case count forecasting (Sklearn blocked by policy)...")
    os.makedirs("forecasting", exist_ok=True)
    
    # Load demographics & cases
    df_demo = pd.read_csv(os.path.join("data", "processed", "tn_demographics.csv"))
    df_cases = pd.read_csv(os.path.join("data", "processed", "tn_cases.csv"))
    
    # Load GNN Embeddings
    with open(os.path.join("data", "processed", "gnn_embeddings.json")) as f:
        embeddings = json.load(f)
        
    X_data = []
    y_data = []
    
    for r_id in df_demo['region_id']:
        r_cases = df_cases[df_cases['region_id'] == r_id].sort_values('date')['cases'].values
        r_demo = df_demo[df_demo['region_id'] == r_id].iloc[0]
        r_emb = embeddings.get(r_id, [0]*16) # 16-dim embedding
        
        # Create rolling window samples
        window = 7
        for i in range(len(r_cases) - window):
            lag_features = r_cases[i:i+window].tolist()
            target = r_cases[i+window]
            
            # Combine lag features, demographics, and GNN embedding
            feature_vector = lag_features + [r_demo['population'], r_demo['density_per_sq_km']] + r_emb
            X_data.append(feature_vector)
            y_data.append(target)
            
    X = np.array(X_data)
    y = np.array(y_data)
    
    # Simple manual train/test split (80/20) since sklearn is blocked
    split_idx = int(0.8 * len(X))
    # Shuffle
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Dataset shape: {X.shape}")
    
    # XGBoost
    xgb = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)
    xgb_rmse = np.sqrt(mse(y_test, xgb_preds))
    xgb_mae = mae(y_test, xgb_preds)
    
    print("\nModel Evaluation:")
    print(f"XGBoost Forecaster - RMSE: {xgb_rmse:.2f}, MAE: {xgb_mae:.2f}")
    
    # Save the model
    xgb.save_model(os.path.join("forecasting", "xgb_model.json"))
    
if __name__ == "__main__":
    train_ensemble()
