import os
import json
import pandas as pd
import numpy as np

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)
    
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Simple Numpy MLP for baseline (PyTorch is blocked in this environment)
class SimpleMLP:
    def __init__(self, input_dim):
        self.W1 = np.random.randn(input_dim, 32) * 0.1
        self.b1 = np.zeros(32)
        self.W2 = np.random.randn(32, 16) * 0.1
        self.b2 = np.zeros(16)
        self.W3 = np.random.randn(16, 1) * 0.1
        self.b3 = np.zeros(1)
        
    def relu(self, x):
        return np.maximum(0, x)
        
    def forward(self, x):
        self.z1 = x @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.relu(self.z2)
        self.out = self.a2 @ self.W3 + self.b3
        return self.out
        
    def backward(self, x, y, lr=0.001):
        m = x.shape[0]
        
        # Gradients
        d_out = 2 * (self.out - y) / m
        
        dW3 = self.a2.T @ d_out
        db3 = np.sum(d_out, axis=0)
        
        d_a2 = d_out @ self.W3.T
        d_z2 = d_a2 * (self.z2 > 0)
        
        dW2 = self.a1.T @ d_z2
        db2 = np.sum(d_z2, axis=0)
        
        d_a1 = d_z2 @ self.W2.T
        d_z1 = d_a1 * (self.z1 > 0)
        
        dW1 = x.T @ d_z1
        db1 = np.sum(d_z1, axis=0)
        
        # Update
        self.W3 -= lr * dW3
        self.b3 -= lr * db3
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

def train_mlp():
    print("Training Numpy MLP baseline for case count forecasting...")
    
    # Load demographics & cases
    df_demo = pd.read_csv(os.path.join("data", "processed", "tn_demographics.csv"))
    df_cases = pd.read_csv(os.path.join("data", "processed", "tn_cases.csv"))
    
    X_data = []
    y_data = []
    
    for r_id in df_demo['region_id']:
        r_cases = df_cases[df_cases['region_id'] == r_id].sort_values('date')['cases'].values
        r_demo = df_demo[df_demo['region_id'] == r_id].iloc[0]
        
        window = 7
        for i in range(len(r_cases) - window):
            lag_features = r_cases[i:i+window].tolist()
            target = r_cases[i+window]
            
            feature_vector = lag_features + [r_demo['population'], r_demo['density_per_sq_km']]
            X_data.append(feature_vector)
            y_data.append(target)
            
    X = np.array(X_data, dtype=np.float32)
    # Normalize features
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X = (X - X_mean) / X_std
    
    y = np.array(y_data, dtype=np.float32).reshape(-1, 1)
    
    # Split
    split_idx = int(0.8 * len(X))
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    model = SimpleMLP(input_dim=X.shape[1])
    
    # Train
    for epoch in range(100):
        preds = model.forward(X_train)
        model.backward(X_train, y_train, lr=0.01)
        
    # Evaluate
    test_preds = model.forward(X_test)
    rmse = np.sqrt(mse(y_test, test_preds))
    mae_val = mae(y_test, test_preds)
    
    print("\nModel Evaluation (MLP Baseline):")
    print(f"MLP Baseline  - RMSE: {rmse:.2f}, MAE: {mae_val:.2f}")
    
    # We could save weights using np.save if needed
    
if __name__ == "__main__":
    train_mlp()
