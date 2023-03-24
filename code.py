import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from pypfopt import expected_returns, risk_models, EfficientFrontier
import plotly.graph_objects as go

# Define function to get stock data from Yahoo Finance
@st.cache
def get_data(tickers):
    df = yf.download(tickers, start="2016-01-01", end="2021-12-31")["Adj Close"]
    return df

# Define function to optimize portfolio and return weights
def optimize_portfolio(df, target_return, risk_tolerance, initial_investment):
    # Calculate expected returns and covariance matrix
    mu = expected_returns.mean_historical_return(df)
    S = risk_models.sample_cov(df)

    # Optimize portfolio
    try:
        ef = EfficientFrontier(mu, S)
        weights = ef.efficient_risk(target_volatility=risk_tolerance, market_neutral=True)
        cleaned_weights = ef.clean_weights()
        cleaned_weights = {ticker: weight for ticker, weight in cleaned_weights.items() if weight > 0}

        # Calculate number of shares for each stock
        prices = df.iloc[-1, :]
        total_value = initial_investment
        shares = {ticker: int((total_value * weight) / prices[ticker]) for ticker, weight in cleaned_weights.items()}

        # Calculate remaining cash and add to stock with highest weight
        remaining_cash = total_value - sum([shares[ticker] * prices[ticker] for ticker in shares])
        max_ticker = max(cleaned_weights, key=cleaned_weights.get)
        shares[max_ticker] += int(remaining_cash / prices[max_ticker])

        # Calculate portfolio performance
        portfolio_returns = (df.pct_change() * shares).sum(axis=1)
        cumulative_returns = (1 + portfolio_returns).cumprod()

        # Visualize portfolio performance
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns.values, name='Portfolio'))
        fig.add_trace(go.Scatter(x=cumulative_returns.index, y=(1 + df.pct_change().mean(axis=1)).cumprod().values, name='Benchmark'))
        fig.update_layout(title='Portfolio Performance', xaxis_title='Date', yaxis_title='Cumulative Returns')
        st.plotly_chart(fig, use_container_width=True)

        return shares

    except Exception as e:
        st.error("Error occurred during optimization: {}".format(str(e)))


# Get user inputs
tickers = st.text_input("Enter comma-separated list of tickers (e.g. AAPL,GOOG,MSFT)", "AAPL,GOOG,MSFT,AMZN,JPM")
initial_investment = st.number_input("Enter initial investment amount", min_value=1)
target_return = st.slider("Select desired return", 0, 30, 10)
risk_tolerance = st.slider("Select risk tolerance (volatility)", 1, 50, 20)

# Get data and optimize portfolio
df = get_data(tickers)
weights = optimize_portfolio(df, target_return/100, risk_tolerance/100, initial_investment)

# Display optimized portfolio weights
if weights is not None:
    st.subheader("Optimized Portfolio")
    for ticker, shares in weights.items():
        st.write(f"{ticker}: {shares} shares")
