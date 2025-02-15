import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load trained model
model = load_model("stock_predictor.h5")

# Streamlit UI
st.title("📈 AI Stock Price Predictor")

# User input for stock symbol
stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA, MSFT):", "AAPL")

# Function to fetch stock data
def get_stock_data(symbol, start="2020-01-01", end="2024-01-01"):
    df = yf.download(symbol, start=start, end=end)
    return df

# Fetch stock data
st.write(f"Fetching data for {stock_symbol}...")
data = get_stock_data(stock_symbol)
st.write(data.tail())

# Prepare Data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data["Close"].values.reshape(-1, 1))

# Create sequences for prediction
def create_sequences(dataset, seq_length=60):
    x = []
    for i in range(seq_length, len(dataset)):
        x.append(dataset[i-seq_length:i, 0])
    return np.array(x)

# Get last 60 days for prediction
last_60_days = scaled_data[-60:]
x_input = create_sequences(last_60_days.reshape(-1, 1))
x_input = np.reshape(x_input, (x_input.shape[0], x_input.shape[1], 1))

# Make prediction
if x_input.shape[0] > 0:
    predicted_price = model.predict(x_input)
    predicted_price = scaler.inverse_transform(predicted_price.reshape(-1, 1))
    st.write(f"📊 **Predicted Stock Price for {stock_symbol}:** ${predicted_price[-1][0]:.2f}")
else:
    st.write("⚠️ Not enough data for prediction!")

# Plot stock price history
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(data.index, data["Close"], label="Actual Price", color="blue")
ax.set_title(f"{stock_symbol} Stock Price History")
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.legend()
st.pyplot(fig)
