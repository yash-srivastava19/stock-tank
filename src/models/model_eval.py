import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import argparse
from models.model_train import LSTMModel

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
    plt.savefig(f"models/{symbol}_prediction_plot.png")

def main(symbol, seq_length):
    test_data = load_test_data(symbol)
    X_test, y_test = prepare_sequences(test_data, seq_length)
    
    # Load the model
    # model = torch.load(f"models/{symbol}_lstm_model.pth")
    model_class = LSTMModel  # Replace with your model class
    model = model_class()  # Instantiate the model
    model.load_state_dict(torch.load(f"models/{symbol}_lstm_model.pth"))
    
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