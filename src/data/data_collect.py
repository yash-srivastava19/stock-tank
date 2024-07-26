import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import argparse

def collect_stock_data(symbol, start_date, end_date):
    """
    Collect stock data for a given symbol and date range.
    """
    stock = yf.Ticker(symbol)
    data = stock.history(start=start_date, end=end_date)
    return data

def save_data(data, filename):
    """
    Save the collected data to a CSV file.
    """
    data.to_csv(f"data/raw/{filename}")

def main(symbol, days):
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    
    data = collect_stock_data(symbol, start_date, end_date)
    save_data(data, f"{symbol}_stock_data.csv")
    print(f"Data for {symbol} has been collected and saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect stock data for a given symbol.")
    parser.add_argument("symbol", type=str, help="Stock symbol (e.g., AAPL)")
    parser.add_argument("--days", type=int, default=365, help="Number of days of historical data to collect")
    args = parser.parse_args()

    main(args.symbol, args.days)