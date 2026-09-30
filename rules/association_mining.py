import os
import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

def run_mining():
    print("Running Association Rule Mining on Risk Factors...")
    os.makedirs("rules", exist_ok=True)
    
    # Load demographics/health data
    df_demo = pd.read_csv(os.path.join("data", "processed", "tn_demographics.csv"))
    df_cases = pd.read_csv(os.path.join("data", "processed", "tn_cases.csv"))
    
    # Define a "surge" (e.g., regions where total cases in window > 75th percentile)
    total_cases = df_cases.groupby('region_id')['cases'].sum().reset_index()
    threshold = total_cases['cases'].quantile(0.75)
    
    surge_regions = total_cases[total_cases['cases'] > threshold]['region_id'].tolist()
    
    # Binarize factors for association mining
    df_demo['high_density'] = df_demo['density_per_sq_km'] > df_demo['density_per_sq_km'].median()
    df_demo['large_population'] = df_demo['population'] > df_demo['population'].median()
    df_demo['low_testing'] = df_demo['testing_rate'] < df_demo['testing_rate'].median()
    df_demo['low_vaccination'] = df_demo['vax_rate'] < df_demo['vax_rate'].median()
    df_demo['had_surge'] = df_demo['region_id'].isin(surge_regions)
    
    # Prepare transaction dataset (only boolean columns)
    cols = ['high_density', 'large_population', 'low_testing', 'low_vaccination', 'had_surge']
    df_trans = df_demo[cols].copy()
    
    # Run FP-Growth
    freq_items = fpgrowth(df_trans, min_support=0.2, use_colnames=True)
    
    # Generate Rules
    rules = association_rules(freq_items, metric="confidence", min_threshold=0.5)
    
    # Filter rules that lead to 'had_surge'
    surge_rules = rules[rules['consequents'] == frozenset({'had_surge'})]
    surge_rules = surge_rules.sort_values('lift', ascending=False)
    
    print(f"\nFound {len(surge_rules)} rules predicting a surge.")
    print(surge_rules[['antecedents', 'support', 'confidence', 'lift']])
    
    surge_rules.to_csv(os.path.join("rules", "surge_rules.csv"), index=False)
    
if __name__ == "__main__":
    run_mining()
