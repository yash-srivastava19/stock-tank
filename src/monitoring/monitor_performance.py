import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from models.model_versioning import load_model_version
from data.data_collect import collect_stock_data
from data.data_preprocess import preprocess_data
from datetime import datetime, timedelta
import json
import os

def calculate_metrics(y_true, y_pred):
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred)
    }

def monitor_performance(symbol, days=30):
    # Load the latest model
    model, metadata = load_model_version(symbol)
    
    # Fetch recent data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    raw_data = collect_stock_data(symbol, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    
    # Preprocess data
    processed_data, _ = preprocess_data(raw_data)
    
    # Prepare sequences (assuming seq_length=10, adjust if different)
    X, y_true = [], []
    seq_length = 10
    for i in range(len(processed_data) - seq_length):
        X.append(processed_data.iloc[i:(i + seq_length)].values)
        y_true.append(processed_data.iloc[i + seq_length]['Target'])
    X = np.array(X)
    y_true = np.array(y_true)
    
    # Make predictions
    y_pred = model.predict(X).flatten()
    
    # Calculate metrics
    current_metrics = calculate_metrics(y_true, y_pred)
    
    # Compare with stored metrics
    stored_metrics = metadata['performance_metrics']
    
    performance_change = {
        key: (current_metrics[key] - stored_metrics[key]) / stored_metrics[key] * 100
        for key in current_metrics
    }
    
    # Save monitoring results
    monitoring_dir = f"monitoring/{symbol}"
    os.makedirs(monitoring_dir, exist_ok=True)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "current_metrics": current_metrics,
        "performance_change": performance_change
    }
    
    with open(f"{monitoring_dir}/performance_{end_date.strftime('%Y%m%d')}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Performance monitoring results for {symbol}:")
    print(json.dumps(results, indent=2))
    
    # You could add alerting logic here, e.g., if performance degrades beyond a certain threshold

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Monitor model performance for a stock symbol")
    parser.add_argument("symbol", type=str, help="Stock symbol (e.g., AAPL)")
    parser.add_argument("--days", type=int, default=30, help="Number of days to monitor")
    args = parser.parse_args()

    monitor_performance(args.symbol, args.days)