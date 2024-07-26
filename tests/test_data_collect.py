import pytest
from src.data.data_collection import collect_stock_data
from datetime import datetime, timedelta

def test_collect_stock_data():
    symbol = "AAPL"
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    data = collect_stock_data(symbol, start_date, end_date)
    
    assert not data.empty, "Data should not be empty"
    assert 'Open' in data.columns, "Data should contain 'Open' column"
    assert 'Close' in data.columns, "Data should contain 'Close' column"
    assert 'High' in data.columns, "Data should contain 'High' column"
    assert 'Low' in data.columns, "Data should contain 'Low' column"
    assert 'Volume' in data.columns, "Data should contain 'Volume' column"