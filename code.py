import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from pypfopt import expected_returns, risk_models, EfficientFrontier
import plotly.graph_objs as go

st.set_page_config(layout='wide')

# Sidebar
st.sidebar.title("Portfolio Optimization")
st.sidebar.subheader("Enter Desired Portfolio Return and Risk")
target_return = st.sidebar.slider("Desired Return", 0.0, 50.0, 10.0, 0.5)
target_risk = st.sidebar.slider("Desired Risk", 0.0, 30.0, 10.0, 0.5)

# Main content
st.title("Portfolio Optimization")
st.markdown("This app helps you optimize your investment portfolio using the Efficient Frontier algorithm.")

# Load stock data
tickers = ["AAPL", "GOOGL", "TSLA", "MSFT", "AMZN"]
df = yf.download(tickers, start="2010-01-01", end="2022-01-01")["Adj Close"]
df = df.dropna()

# Calculate expected returns and covariance matrix
mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)

# Optimize portfolio
try:
    ef = EfficientFrontier(mu, S)
    weights = ef.efficient_return(target_return/100, market_neutral=True)
    cleaned_weights = ef.clean_weights()
    ef.portfolio_performance(verbose=True)
    # Visualize portfolio performance
    portfolio_returns = (df.pct_change() * cleaned_weights).sum(axis=1)
    cumulative_returns = (1 + portfolio_returns).cumprod()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns.values, name='Portfolio'))
    fig.add_trace(go.Scatter(x=cumulative_returns.index, y=(1 + df.pct_change().mean(axis=1)).cumprod().values, name='Benchmark'))
    fig.update_layout(title='Portfolio Performance', xaxis_title='Date', yaxis_title='Cumulative Returns')
    st.plotly_chart(fig, use_container_width=True)

    # Display optimized portfolio weights
    st.subheader("Optimized Portfolio Weights")
    st.write(cleaned_weights * 100)

except Exception as e:
    st.error("Error occurred during optimization: {}".format(str(e)))

