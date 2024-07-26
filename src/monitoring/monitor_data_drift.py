import pandas as pd
import numpy as np
from scipy import stats
from data.data_collect import collect_stock_data
from datetime import datetime, timedelta
import json
import os

def detect_drift(reference_data, current_data, threshold=0.05):
    drift_results = {}
    for column in reference_data.columns:
        _, p_value = stats.ks_2samp(reference_data[column], current_data[column])
        drift_detected = p_value < threshold
        drift_results[column] = {
            "p_value": p_value,
            "drift_detected": drift_detected
        }
    return drift_results

def monitor_data_drift(symbol, reference_days=365, current_days=30):
    end_date = datetime.now()
    reference_start_date = end_date - timedelta(days=reference_days)
    current_start_date = end_date - timedelta(days=current_days)
    
    # Fetch reference and current data
    reference_data = collect_stock_data(symbol, reference_start_date.strftime("%Y-%m-%d"), current_start_date.strftime("%Y-%m-%d"))
    current_data = collect_stock_data(symbol, current_start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    
    # Detect drift
    drift_results = detect_drift(reference_data, current_data)
    
    # Save drift detection results
    monitoring_dir = f"monitoring/{symbol}"
    os.makedirs(monitoring_dir, exist_ok=True)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "drift_results": drift_results
    }
    
    with open(f"{monitoring_dir}/drift_{end_date.strftime('%Y%m%d')}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Data drift detection results for {symbol}:")
    print(json.dumps(results, indent=2))
    
    # You could add alerting logic here, e.g., if drift is detected in multiple features

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Detect data drift for a stock symbol")
    parser.add_argument("symbol", type=str, help="Stock symbol (e.g., AAPL)")
    parser.add_argument("--reference_days", type=int, default=365, help="Number of days for reference data")
    parser.add_argument("--current_days", type=int, default=30, help="Number of days for current data")
    args = parser.parse_args()

    monitor_data_drift(args.symbol, args.reference_days, args.current_days)