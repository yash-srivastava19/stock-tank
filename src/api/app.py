import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import torch
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler

def get_stock_data(symbol, start_date, end_date):
    """
    Fetch stock data from Yahoo Finance.
    """
    stock = yf.Ticker(symbol)
    data = stock.history(start=start_date, end=end_date)
    
    data['Target'] = data['Close'].shift(-1)
    data = data.dropna()

    return data[['Open', 'High', 'Low', 'Close', 'Volume', 'Target']]

def preprocess_data(data):
    """
    Preprocess the input data.
    """
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = data[features]
    
    scaler = MinMaxScaler()
    data[features] = scaler.fit_transform(data[features])
    
    data['Target'] = data['Close'].shift(-1)
    data = data.dropna()
    
    return data, scaler

def prepare_sequences(data, seq_length):
    """
    Prepare sequences for LSTM model.
    """
    X = []
    for i in range(len(data) - seq_length + 1):
        X.append(data.iloc[i:i + seq_length].values)
    return np.array(X)

def predict_price(model, data):
    """
    Make a price prediction using the LSTM model.
    """
    data_tensor = torch.tensor(data, dtype=torch.float32)
    with torch.no_grad():
        prediction = model(data_tensor)
    return prediction.numpy()

st.title("Stock Price Prediction App")
st.markdown("""
    This app predicts the stock prices for the next few days using a pre-trained LSTM model.
    Select a stock symbol and the number of days to predict.
""")


symbol = st.selectbox("Select stock symbol:", ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"])
days = st.number_input("Enter number of days to predict:", min_value=1, max_value=30, value=7)

with st.spinner("Loading model..."):
    model = torch.jit.load(f"models/versions/{symbol}/v1_model.pth")

model.eval()

#days = st.number_input("Enter number of days to predict:", min_value=1, max_value=30, value=7)

if st.button("Predict"):
    # Fetch historical data
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=100)  # We'll use 100 days of historical data
    data = get_stock_data(symbol, start_date, end_date)
    
    # Preprocess data
    data_preprocessed, scaler = preprocess_data(data)
    
    # Prepare sequences
    seq_length = 10  # Make sure this matches the expected input sequence length for your model
    sequences = prepare_sequences(data_preprocessed, seq_length)
    
    # Make predictions for the specified number of days
    predictions = []
    current_sequence = sequences[-1]  # Start with the last sequence from historical data
    
    for _ in range(days):
        current_sequence = current_sequence.reshape(1, seq_length, -1)  # Reshape for model input
        
        prediction = predict_price(model, current_sequence)
        prediction_original_scale = scaler.inverse_transform([[0, 0, 0, prediction[0, 0], 0]])[0, 3]
        
        predictions.append(prediction_original_scale)
        
        # Update the sequence for the next prediction
        new_data = np.array([[
            current_sequence[0, -1, 0],  # Open
            current_sequence[0, -1, 1],  # High
            current_sequence[0, -1, 2],  # Low
            current_sequence[0, -1, 3],  # Close
            data_preprocessed['Volume'].iloc[-1],  # Volume (unchanged)
            current_sequence[0, -1, 4]   # Target
        ]])
        new_data = np.expand_dims(new_data, axis=1)
        
        current_sequence = np.append(current_sequence[:, 1:, :], new_data, axis=1)
    

    # Display predictions
    st.write(f"Predicted stock prices for the next {days} days:")
    
    import pandas as pd

    # Create a DataFrame for the predictions
    predictions_df = pd.DataFrame({
        'Day': range(1, days + 1),
        'Predicted Price': [f"${price:.2f}" for price in predictions]
    })

    # Display the DataFrame as a table
    st.table(predictions_df)


    st.write("Predicted data:")
    # x should be a dataframindex
    # use only the days, not AM,PM
    x = pd.date_range(start=end_date, periods=days)
    x = x.strftime('%d-%m')

    st.line_chart(pd.DataFrame({
        'Date': x, 
        'Predicted Price': predictions
    }).set_index('Date'), x_label='Date', y_label='Price', use_container_width=True)


st.write("Note: These predictions are based on historical data and should not be used as financial advice.")