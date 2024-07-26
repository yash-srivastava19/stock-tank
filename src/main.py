import argparse
import schedule
import time
import torch
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime, timedelta
from data.data_collect import collect_stock_data
from models.model_train import LSTMModel, prepare_sequences, train_model
from data.data_preprocess import preprocess_data, split_data
from models.model_train import prepare_sequences, train_model
from models.model_eval import evaluate_model
from models.model_versioning import save_model_version, load_model_version
from models.model_autoretrain import retrain_model
from monitoring.monitor_performance import monitor_performance
from monitoring.monitor_data_drift import monitor_data_drift
from utils.logging_utils import main_logger, send_alert

def initial_training(symbol, days=365, seq_length=10):
    main_logger.info(f"Starting initial training for {symbol}")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Collect data
    raw_data = collect_stock_data(symbol, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    
    # Preprocess data
    processed_data, scaler = preprocess_data(raw_data)
    train_data, test_data = split_data(processed_data)
    
    # Prepare sequences
    X_train, y_train = prepare_sequences(train_data, seq_length)
    X_test, y_test = prepare_sequences(test_data, seq_length)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    input_size = X_train.shape[2]
    hidden_size = 50
    output_dim = 1
    dropout = 0.2
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel(input_size, hidden_size, output_dim, dropout)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    # Train model
    # print("Reached Here")
    train_model(model, train_loader, test_loader, device)
    
    # Evaluate model
    _, mse, mae, r2 = evaluate_model(model, X_test, y_test, device)
    
    # Save model
    performance_metrics = {"mse": mse, "mae": mae, "r2": r2}
    version = save_model_version(symbol, model, performance_metrics)
    
    main_logger.info(f"Initial training completed for {symbol}. Model version: {version}")

def run_daily_tasks(symbol):
    main_logger.info(f"Running daily tasks for {symbol}")
    
    # Monitor performance
    monitor_performance(symbol)
    
    # Check for data drift
    monitor_data_drift(symbol)
    
    main_logger.info(f"Daily tasks completed for {symbol}")

def run_weekly_tasks(symbol):
    main_logger.info(f"Running weekly tasks for {symbol}")
    
    # Retrain model
    retrain_model(symbol)
    
    main_logger.info(f"Weekly tasks completed for {symbol}")

def schedule_tasks(symbol):
    schedule.every().day.at("00:00").do(run_daily_tasks, symbol)
    schedule.every().monday.at("01:00").do(run_weekly_tasks, symbol)

    while True:
        schedule.run_pending()
        time.sleep(3600)  # Sleep for an hour between checks

def main(symbol):
    try:
        # Check if model exists, if not perform initial training
        try:
            model, metadata = load_model_version(symbol)
            main_logger.info(f"Existing model found for {symbol}")
        except FileNotFoundError:
            main_logger.info(f"No existing model found for {symbol}. Performing initial training.")
            initial_training(symbol)

        # Schedule regular tasks
        #schedule_tasks(symbol)
    
    except Exception as e:
        print(e)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run the stock prediction MLOps pipeline")
    parser.add_argument("symbol", type=str, help="Stock symbol (e.g., AAPL)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    args = parser.parse_args()

    main(args.symbol)