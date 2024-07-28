from datetime import datetime, timedelta
from models.model_autoretrain import retrain_model
from monitoring.monitor_performance import monitor_performance
from monitoring.monitor_data_drift import monitor_data_drift
from utils.logging_utils import main_logger, send_alert
from data.data_collect import collect_stock_data, save_data
from data.data_preprocess import preprocess_data, split_data, save_processed_data

def get_and_save_fresh_data(symbol, days=365):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    print(f"Getting fresh data for {symbol}")
    data = collect_stock_data(symbol, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    save_data(data, f"{symbol}_stock_data.csv")

    # Preprocess data
    processd_data, scaler = preprocess_data(data)
    train_data, test_data = split_data(processd_data)
    save_processed_data(train_data, test_data, scaler, symbol)

def run_daily_tasks(symbol):
    # Does it make sense to retrain_everyday? No
    main_logger.info(f"Running daily tasks for {symbol}")
    
    # Monitor performance
    monitor_performance(symbol, freq="daily")
    
    # Check for data drift
    monitor_data_drift(symbol, freq="daily")
    
    main_logger.info(f"Daily tasks completed for {symbol}")

def run_weekly_tasks(symbol):
    main_logger.info(f"Running weekly tasks for {symbol}")
    
    # Retrain model
    retrain_model(symbol)

    # Monitor performance
    monitor_performance(symbol, freq="weekly")

    # Check for data drift
    monitor_data_drift(symbol, freq="weekly")
    
    main_logger.info(f"Weekly tasks completed for {symbol}")

def run_monthly_tasks(symbol):
    main_logger.info(f"Running monthly tasks for {symbol}")
    # Replace with new_batch_data

    # Retrain model
    retrain_model(symbol)
    
    # Monitor performance
    monitor_performance(symbol, freq="monthly")
    
    # Check for data drift
    monitor_data_drift(symbol, freq="monthly")
    
    main_logger.info(f"Monthly tasks completed for {symbol}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the stock prediction MLOps pipeline")
    
    # First, we'll accept either dail/weekly/monthy as arg to run tasks accordingly.
    parser.add_argument("--freq", type=str, help="Task to run: daily, weekly, monthly")
    args = parser.parse_args()

    if args.freq == "daily":
        for symbol in ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]:
            run_daily_tasks(symbol)

    elif args.freq == "weekly":
        for symbol in ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]:
            run_weekly_tasks(symbol)
    
    elif args.freq == "monthly":
        for symbol in ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]:
            run_monthly_tasks(symbol)
        
    else:
        main_logger.error("Invalid task. Please specify daily, weekly, or monthly.")
        send_alert("Invalid task specified. Please specify daily, weekly, or monthly.")
                            