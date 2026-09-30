import os
import json
import urllib.request
import pandas as pd
from datetime import datetime, timedelta

def fetch_and_clean():
    print("Fetching Covid19India data...")
    
    # URLs for the archived COVID-19 India API
    # URLs for the archived COVID-19 India CSV API
    csv_url = "https://data.covid19india.org/csv/latest/districts.csv"
    data_url = "https://data.covid19india.org/v4/min/data.min.json"
    
    import requests
    import io
    
    # Load Population JSON
    response_data = requests.get(data_url)
    data_json = response_data.json()
        
    # Load Timeseries CSV
    response_csv = requests.get(csv_url)
    df_raw = pd.read_csv(io.StringIO(response_csv.text))
    
    # Focus on Tamil Nadu (TN)
    state_code = "TN"
    state_name = "Tamil Nadu"
    df_raw = df_raw[df_raw['State'] == state_name]
    
    # We will look at data from August to October 2021
    df_raw['Date'] = pd.to_datetime(df_raw['Date'])
    start_date = pd.to_datetime('2021-08-01')
    end_date = pd.to_datetime('2021-10-31')
    
    df_raw = df_raw[(df_raw['Date'] >= start_date) & (df_raw['Date'] <= end_date)]

    records = []
    demographics = []

    # Real area for a subset of major districts to calculate density
    areas_sq_km = {
        "Chennai": 426,
        "Coimbatore": 4723,
        "Madurai": 3741,
        "Salem": 5205,
        "Tiruchirappalli": 4404,
        "Vellore": 2080,
        "Erode": 5722,
        "Tirunelveli": 3907,
        "Thanjavur": 3396,
        "Thoothukkudi": 4621
    }

    print("Processing district timeseries and demographics...")
    
    districts = df_raw['District'].unique()
    
    for dist in districts:
        if dist == "Unknown":
            continue
            
        # Get demographics & health factors
        meta = data_json.get(state_code, {}).get('districts', {}).get(dist, {}).get('meta', {})
        total = data_json.get(state_code, {}).get('districts', {}).get(dist, {}).get('total', {})
        
        population = meta.get('population', None)
        if population is None:
            continue
            
        area = areas_sq_km.get(dist, 4000)
        density = population / area
        
        # Real epidemiological features (total as of late 2021)
        tested = total.get('tested', 0)
        vaccinated1 = total.get('vaccinated1', 0)
        
        testing_rate = tested / population if population else 0
        vax_rate = vaccinated1 / population if population else 0
        
        region_id = f"{state_code}-{dist[:3].upper()}"
        
        # Add demographics
        demographics.append({
            "region_id": region_id,
            "district": dist,
            "population": population,
            "density_per_sq_km": round(density, 2),
            "testing_rate": round(testing_rate, 4),
            "vax_rate": round(vax_rate, 4)
        })

        # Get timeseries
        dist_df = df_raw[df_raw['District'] == dist]
        # The CSV has cumulative cases ("Confirmed"), we need daily new cases
        dist_df = dist_df.sort_values('Date')
        dist_df['cases'] = dist_df['Confirmed'].diff().fillna(0)
        dist_df['cases'] = dist_df['cases'].apply(lambda x: max(0, x)) # No negative cases
        
        for _, row in dist_df.iterrows():
            records.append({
                "date": row['Date'].strftime("%Y-%m-%d"),
                "region_id": region_id,
                "district": dist,
                "cases": int(row['cases'])
            })

    df_cases = pd.DataFrame(records)
    df_demo = pd.DataFrame(demographics)

    # Make output dirs
    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    
    cases_path = os.path.join("data", "processed", "tn_cases.csv")
    demo_path = os.path.join("data", "processed", "tn_demographics.csv")
    
    df_cases.to_csv(cases_path, index=False)
    df_demo.to_csv(demo_path, index=False)
    
    print(f"Saved {len(df_cases)} case records to {cases_path}")
    print(f"Saved {len(df_demo)} demographic records to {demo_path}")
    print("\nSample Data:")
    print(df_cases.head())
    print(df_demo.head())

if __name__ == "__main__":
    fetch_and_clean()
