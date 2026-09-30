import os
import json
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

def export_data():
    print("Exporting region_risk_output.json per shared data contract...")
    
    # Load required data
    df_demo = pd.read_csv(os.path.join("data", "processed", "tn_demographics.csv"))
    df_cases = pd.read_csv(os.path.join("data", "processed", "tn_cases.csv"))
    
    with open(os.path.join("data", "processed", "gnn_embeddings.json")) as f:
        embeddings = json.load(f)
        
    xgb = XGBRegressor()
    xgb.load_model(os.path.join("forecasting", "xgb_model.json"))
    
    # Load surge rules to assign contributing factors
    try:
        rules_df = pd.read_csv(os.path.join("rules", "surge_rules.csv"))
        # Get unique antecedents
        all_factors = set()
        for antecedents in rules_df['antecedents']:
            # Example parse: "frozenset({'high_density', 'low_vaccination'})"
            clean_str = antecedents.replace("frozenset({", "").replace("})", "").replace("'", "")
            factors = [f.strip() for f in clean_str.split(",")]
            all_factors.update(factors)
    except Exception:
        all_factors = []
        
    timestep = "2026-09-01" # Target simulation start date as per schema
    regions_output = []
    
    for r_id in df_demo['region_id']:
        r_cases = df_cases[df_cases['region_id'] == r_id].sort_values('date')['cases'].values
        r_demo = df_demo[df_demo['region_id'] == r_id].iloc[0]
        r_emb = embeddings.get(r_id, [0]*16)
        
        # Prepare input for predicting next day's cases
        lag_features = r_cases[-7:].tolist()
        feature_vector = lag_features + [r_demo['population'], r_demo['density_per_sq_km']] + r_emb
        
        # Predict cases
        pred_cases = float(xgb.predict(np.array([feature_vector]))[0])
        pred_cases = max(0, pred_cases) # No negative cases
        
        # Calculate Risk Score (0-1) based on predicted cases & population
        risk_score = min(1.0, pred_cases / (r_demo['population'] * 0.0001 + 1)) # arbitrary scaling for realistic risk
        
        # Determine contributing factors based on rules + region characteristics
        region_factors = []
        if r_demo['density_per_sq_km'] > df_demo['density_per_sq_km'].median():
            region_factors.append("high_density")
        if r_demo['population'] > df_demo['population'].median():
            region_factors.append("large_population")
        if r_demo['vax_rate'] < df_demo['vax_rate'].median():
            region_factors.append("low_vaccination")
        if r_demo['testing_rate'] < df_demo['testing_rate'].median():
            region_factors.append("low_testing")
            
        regions_output.append({
            "region_id": r_id,
            "risk_score": round(risk_score, 4),
            "predicted_cases": int(round(pred_cases)),
            "embedding": [round(e, 4) for e in r_emb],
            "contributing_factors": region_factors
        })
        
    output_schema = {
        "timestep": timestep,
        "regions": regions_output
    }
    
    out_path = "region_risk_output.json"
    with open(out_path, "w") as f:
        json.dump(output_schema, f, indent=2)
        
    print(f"Exported data to {out_path} with {len(regions_output)} regions.")
    
if __name__ == "__main__":
    export_data()
