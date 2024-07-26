import schedule
import time
from datetime import datetime, timedelta
import yfinance as yf

from data.data_preprocess import preprocess_data, split_data
from models.model_train import prepare_sequences, train_model
from models.model_eval import evaluate_model
from models.model_versioning import save_model_version, load_model_version
import pandas as pd
import numpy as np

def fetch_new_data(symbol, days=30):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    stock = yf.Ticker(symbol)
    data = stock.history(start=start_date, end=end_date)
    return data[['Open', 'High', 'Low', 'Close', 'Volume']]

def retrain_model(symbol, seq_length=10):
    print(f"Retraining model for {symbol}")
    
    # Load the latest model and its metadata
    try:
        _, metadata = load_model_version(symbol)
        last_train_date = datetime.fromisoformat(metadata['timestamp'])
    except FileNotFoundError:
        last_train_date = datetime.now() - timedelta(days=365)  # If no model exists, use 1 year of data
    
    # Fetch new data since last training
    new_data = fetch_new_data(symbol, (datetime.now() - last_train_date).days)
    
    # Preprocess new data
    processed_data, _ = preprocess_data(new_data)
    
    # Prepare sequences
    X, y = prepare_sequences(processed_data, seq_length)
    
    # Train model
    model, _ = train_model(X, y, X, y)  # Using all data for training in this example
    
    # Evaluate model
    _, mse, mae, r2 = evaluate_model(model, X, y)
    
    # Save new model version
    performance_metrics = {
        "mse": mse,
        "mae": mae,
        "r2": r2
    }
    new_version = save_model_version(symbol, model, performance_metrics)
    
    print(f"Model for {symbol} retrained and saved as version {new_version}")

def schedule_retraining(symbol, interval_days=7):
    schedule.every(interval_days).days.do(retrain_model, symbol)

    while True:
        schedule.run_pending()
        time.sleep(3600)  # Sleep for an hour between checks

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Schedule automated retraining for a stock symbol")
    parser.add_argument("symbol", type=str, help="Stock symbol (e.g., AAPL)")
    parser.add_argument("--interval", type=int, default=7, help="Retraining interval in days")
    args = parser.parse_args()

    schedule_retraining(args.symbol, args.interval)