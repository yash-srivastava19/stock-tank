import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import argparse

def load_data(symbol):
    """
    Load data from a CSV file.
    """
    return pd.read_csv(f"data/raw/{symbol}_stock_data.csv", index_col=0, parse_dates=True)

def preprocess_data(data):
    """
    Preprocess the data for model training.
    """
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = data[features]
    
    scaler = MinMaxScaler()
    data[features] = scaler.fit_transform(data[features])

    data['Target'] = data['Close'].shift(-1)
    data = data.dropna()
    
    return data, scaler

def split_data(data, test_size=0.2):
    """
    Split the data into training and testing sets.
    """
    split_idx = int(len(data) * (1 - test_size))
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    return train_data, test_data

def save_processed_data(train_data, test_data, scaler, symbol):
    """
    Save the processed data and scaler.
    """
    train_data.to_csv(f"data/processed/{symbol}_train_data.csv")
    test_data.to_csv(f"data/processed/{symbol}_test_data.csv")
    np.save(f"data/processed/{symbol}_scaler.npy", scaler.scale_)

def main(symbol):
    raw_data = load_data(symbol)
    processed_data, scaler = preprocess_data(raw_data)
    train_data, test_data = split_data(processed_data)
    save_processed_data(train_data, test_data, scaler, symbol)
    print(f"Data for {symbol} has been preprocessed and saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess stock data for a given symbol.")
    parser.add_argument("symbol", type=str, help="Stock symbol (e.g., AAPL)")
    args = parser.parse_args()

    main(args.symbol)