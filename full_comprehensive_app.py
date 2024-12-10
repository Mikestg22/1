
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import date
import plotly.graph_objects as go

# Define top 50 stocks
top_50_stocks = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "BRK.B", "V", "JPM",
    "JNJ", "WMT", "PG", "UNH", "HD", "MA", "DIS", "PYPL", "BAC", "NFLX",
    "ADBE", "KO", "PEP", "NKE", "T", "PFE", "MRK", "INTC", "XOM", "CSCO",
    "ORCL", "CMCSA", "ABT", "CRM", "COST", "VZ", "ACN", "AMD", "UPS", "QCOM",
    "TXN", "AVGO", "LIN", "LOW", "HON", "AMGN", "CVX", "MDT", "PM", "NEE"
]

# Initialize session state
if "stock_data" not in st.session_state:
    st.session_state.stock_data = None
if "selected_stock" not in st.session_state:
    st.session_state.selected_stock = None

# Function to fetch stock data
@st.cache_data
def fetch_stock_data(ticker, start_date, end_date):
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        if stock_data.empty:
            st.error(f"No data found for ticker '{ticker}'.")
            return None
        return stock_data
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return None

# Function to predict price movement
def predict_price_movement(data, days=7):
    if len(data) < 2:
        return None, None
    data['Days'] = np.arange(len(data))
    X = data[['Days']].values
    y = data['Close'].values
    model = LinearRegression()
    model.fit(X, y)
    future_days = np.arange(len(data), len(data) + days).reshape(-1, 1)
    predictions = model.predict(future_days)
    return data['Close'].iloc[-1], predictions[-1]

# Function to display candlestick chart
def display_candlestick_chart(data):
    st.subheader("Candlestick Chart")
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data["Open"],
        high=data["High"],
        low=data["Low"],
        close=data["Close"]
    )])
    fig.update_layout(title="Candlestick Chart", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig)

# Function to analyze options and make recommendations
def analyze_options(stock, predicted_price, current_price):
    st.subheader("Options Recommendations")
    try:
        if not stock.options:
            st.error("No options data available for this stock.")
            return

        # Fetch options data
        expiry_dates = stock.options
        expiry = st.selectbox("Select Expiry Date", expiry_dates)
        options_chain = stock.option_chain(expiry)
        calls = options_chain.calls
        puts = options_chain.puts

        # Display options data
        st.write("Call Options")
        st.dataframe(calls)
        st.write("Put Options")
        st.dataframe(puts)

        # Recommendations
        if predicted_price > current_price * 1.05:
            st.success(
                f"Recommendation: BUY CALL OPTIONS - The stock price is predicted to rise significantly from ${current_price:.2f} to ${predicted_price:.2f}. "
                "Call options allow you to profit from this upward movement."
            )
        elif predicted_price < current_price * 0.95:
            st.success(
                f"Recommendation: BUY PUT OPTIONS - The stock price is predicted to drop significantly from ${current_price:.2f} to ${predicted_price:.2f}. "
                "Put options allow you to profit from this downward movement."
            )
        else:
            st.info(
                f"Recommendation: HOLD - The stock price is predicted to stay around ${predicted_price:.2f}. "
                "Options trading may not be profitable in this scenario."
            )
    except Exception as e:
        st.error(f"Error analyzing options: {e}")

# Placeholder for economic events
def fetch_economic_events():
    st.subheader("Upcoming Economic Events")
    st.info("Economic events such as earnings, Fed announcements, and other market-moving events will be displayed here.")

# Placeholder for news sentiment analysis
def fetch_news_sentiment(ticker):
    st.subheader("News Sentiment Analysis")
    st.info(f"Fetching news sentiment for {ticker}... (This feature is under development)")

# App layout with tabs
tabs = st.tabs(["Stock Analysis", "Options Analysis", "Economic Events", "News Sentiment"])

# Tab 1: Stock Analysis
with tabs[0]:
    st.header("Stock Analysis")
    selected_stock = st.selectbox("Select a Stock", top_50_stocks)
    start_date = st.date_input("Start Date", value=pd.to_datetime("2022-01-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("2024-01-01"))

    if st.button("Analyze Stock"):
        st.session_state.selected_stock = selected_stock
        stock_data = fetch_stock_data(selected_stock, start_date, end_date)
        if stock_data is not None:
            st.session_state.stock_data = stock_data
            st.write("Stock Data", stock_data.tail(10))
            display_candlestick_chart(stock_data)
            current_price, predicted_price = predict_price_movement(stock_data)
            if current_price is not None and predicted_price is not None:
                st.write(f"Current Price: ${current_price:.2f}")
                st.write(f"Predicted Price: ${predicted_price:.2f}")

# Tab 2: Options Analysis
with tabs[1]:
    st.header("Options Analysis")
    if st.session_state.selected_stock:
        st.write(f"Fetching options for {st.session_state.selected_stock}...")
        stock = yf.Ticker(st.session_state.selected_stock)
        if st.session_state.stock_data is not None:
            current_price, predicted_price = predict_price_movement(st.session_state.stock_data)
            analyze_options(stock, predicted_price, current_price)
    else:
        st.info("Select a stock in the Stock Analysis tab first.")

# Tab 3: Economic Events
with tabs[2]:
    fetch_economic_events()

# Tab 4: News Sentiment
with tabs[3]:
    if st.session_state.selected_stock:
        fetch_news_sentiment(st.session_state.selected_stock)
    else:
        st.info("Select a stock in the Stock Analysis tab first.")
