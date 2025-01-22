import streamlit as st
import yfinance as yf
from PIL import Image
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd

# Page layout
st.set_page_config(layout="wide", page_title="Stock Analysis Dashboard")

# Create a title for the dashboard
st.title("Stock Analysis Dashboard")


# Function to get stock info
def get_stock_info(ticker):
    stock = yf.Ticker(ticker)
    return stock.info


# Function to get historical data
def get_historical_data(ticker, period="1y"):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    return hist


# Create a sidebar for user input
st.sidebar.header("Stock Analysis")
ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL")

if ticker:
    # Get stock info and historical data
    stock_info = get_stock_info(ticker)
    hist_data = get_historical_data(ticker)

    # Display basic stock information
    st.subheader("Stock Information")
    st.write(f"Current Price: ${stock_info['currentPrice']}")
    st.write(f"Market Cap: ${stock_info['marketCap']:,.0f}")
    st.write(f"Sector: {stock_info['sector']}")

    # Historical price chart
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=hist_data.index,
                open=hist_data["Open"],
                high=hist_data["High"],
                low=hist_data["Low"],
                close=hist_data["Close"],
            )
        ]
    )

    st.subheader("Historical Price Chart")
    st.plotly_chart(fig, use_container_width=True)

    # Technical Indicators section
    st.header("Technical Indicators")

    with st.expander("Moving Averages"):
        ma_window = st.slider("Window Size", 10, 200, value=50)

        # Calculate moving average
        hist_data["MA"] = hist_data["Close"].rolling(window=ma_window).mean()

        # Create a new figure for MA
        ma_fig = go.Figure(
            data=[
                go.Scatter(x=hist_data.index, y=hist_data["Close"], name="Close Price"),
                go.Scatter(
                    x=hist_data.index, y=hist_data["MA"], name=f"{ma_window} Day Moving Average"
                ),
            ]
        )

        # Update layout
        ma_fig.update_layout(
            title=f"{ticker} Closing Price & {ma_window} Day MA",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
        )

        st.plotly_chart(ma_fig, use_container_width=True)

    with st.expander("RSI"):
        rsi_window = st.slider("RSI Window", 10, 100, value=14)

        # Calculate RSI
        delta = hist_data["Close"].diff().dropna()
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0

        roll_up = up.ewm(com=rsi_window - 1, adjust=False).mean()
        roll_down = down.ewm(com=rsi_window - 1, adjust=False).mean().abs()

        RS = roll_up / roll_down
        RSI = 100.0 - (100.0 / (1.0 + RS))

        # Create a new figure for RSI
        rsi_fig = go.Figure(
            data=[go.Scatter(x=hist_data.index[-rsi_window:], y=RSI[rsi_window:], name="RSI")]
        )

        # Add horizontal lines for RSI levels
        rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")
        rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")

        # Update layout
        rsi_fig.update_layout(
            title=f"{ticker} Relative Strength Index ({rsi_window} period)",
            xaxis_title="Date",
            yaxis_title="RSI",
            yaxis=dict(range=[0, 100]),
        )

        st.plotly_chart(rsi_fig, use_container_width=True)

else:
    st.warning("Please enter a valid stock ticker symbol.")

# Add about section
st.markdown("---")
st.header("About this Dashboard")
st.write(
    "This dashboard provides comprehensive analysis of stock performance including historical price data, key metrics, and technical indicators."
)

# Run the app
# if __name__ == "__main__":
#     st.run()
