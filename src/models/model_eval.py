import os
import json
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def get_latest_model_version(symbol):
    versions_dir = f"models/versions/{symbol}"
    if not os.path.exists(versions_dir):
        return None
    
    versions = [int(f.split('_')[0][1:]) for f in os.listdir(versions_dir) if f.endswith('_model.pth')]
    if not versions:
        return None
    
    latest_version = max(versions)
    return latest_version

def load_model_version(symbol, version=None):
    if version is None:
        version = get_latest_model_version(symbol)
        if version is None:
            raise FileNotFoundError(f"No model versions found for {symbol}")
    
    model_path = f"models/versions/{symbol}/v{version}_model.pth"
    metadata_path = f"models/versions/{symbol}/v{version}_metadata.json"
    
    if not os.path.exists(model_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Model version {version} not found for {symbol}")
    
    
    model = torch.jit.load(model_path)
    model.eval()  # Set the model to evaluation mode
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return model, metadata


def load_test_data(symbol):
    """
    Load the processed test data.
    """
    return pd.read_csv(f"data/processed/{symbol}_test_data.csv", index_col=0, parse_dates=True)

def prepare_sequences(data, seq_length):
    """
    Prepare sequences for LSTM model.
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data.iloc[i:(i + seq_length)].values)
        y.append(data.iloc[i + seq_length]['Target'])
    return np.array(X), np.array(y)

def evaluate_model(model, X_test, y_test, device):
    """
    Evaluate the model and return predictions and metrics.
    """
    model.to(device)
    model.eval()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        predictions = model(X_test_tensor).cpu().numpy().flatten()
    
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    return predictions, mse, mae, r2

def plot_results(y_test, predictions, symbol):
    """
    Plot actual vs predicted values.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.title(f"{symbol} Stock Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.savefig(f"plots/{symbol}_prediction_plot.png")

def main(symbol, seq_length):
    test_data = load_test_data(symbol)
    X_test, y_test = prepare_sequences(test_data, seq_length)
    
    # Load the model
    # model = torch.load(f"models/{symbol}_lstm_model.pth")
    try:
        model, metadata = load_model_version(symbol)
        #main_logger.info(f"Existing model found for {symbol}")
    except FileNotFoundError:
        model, version = initial_training(symbol) # FIXME: This function is not defined
        print(f"No existing model found for {symbol}.")
    
    # Evaluate the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    predictions, mse, mae, r2 = evaluate_model(model, X_test, y_test, device)
    
    print(f"Evaluation results for {symbol}:")
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R2 Score: {r2}")
    
    plot_results(y_test, predictions, symbol)
    print(f"Prediction plot saved as plots/{symbol}_prediction_plot.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LSTM model for stock price prediction.")
    parser.add_argument("symbol", type=str, help="Stock symbol (e.g., AAPL)")
    parser.add_argument("--seq_length", type=int, default=10, help="Sequence length for LSTM input")
    args = parser.parse_args()

    main(args.symbol, args.seq_length)