import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from pypfopt import expected_returns, risk_models, EfficientFrontier
from datetime import datetime, timedelta

# Set start and end date for historical stock data
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=365*5)).strftime('%Y-%m-%d')

# Get list of S&P 500 companies
table=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
df = table[0]

# Allow user to input initial investment, desired return, and risk tolerance
initial_investment = st.number_input("Enter your initial investment:", min_value=1)
desired_return = st.slider("Enter your desired annual return:", min_value=0, max_value=100, step=1)
risk_tolerance = st.slider("Enter your risk tolerance (lower value means lower risk):", min_value=0, max_value=100, step=1)

# Get historical data for selected stocks
symbols = st.multiselect("Select the companies you want to include in your portfolio:", df['Symbol'].values)
data = yf.download(symbols, start=start_date, end=end_date)['Adj Close']
data.dropna(inplace=True)

# Calculate expected returns and covariance matrix
mu = expected_returns.mean_historical_return(data)
S = risk_models.sample_cov(data)

# Optimize portfolio
try:
    ef = EfficientFrontier(mu, S)
    weights = ef.efficient_return(desired_return/100)
    weights = ef.clean_weights()
    allocation = {symbol: int(weights[symbol] * initial_investment / data[symbol][-1]) for symbol in weights}
    portfolio_value = sum(allocation[symbol] * data[symbol][-1] for symbol in allocation)
    returns = data.pct_change().dropna()
    portfolio_returns = (returns * pd.Series(allocation)).sum(axis=1)
    annualized_return = ((1 + portfolio_returns.mean()) ** 252 - 1)
    annualized_volatility = (portfolio_returns.std() * np.sqrt(252))
    sharpe_ratio = (annualized_return - 0.02) / annualized_volatility
    
    # Show allocation of shares
    st.subheader("Allocation of Shares")
    for symbol, shares in allocation.items():
        st.write("{}: {}".format(symbol, shares))

    # Show portfolio statistics
    st.subheader("Portfolio Statistics")
    st.write("Portfolio Value: ${:,.2f}".format(portfolio_value))
    st.write("Annualized Return: {:.2%}".format(annualized_return))
    st.write("Annualized Volatility: {:.2%}".format(annualized_volatility))
    st.write("Sharpe Ratio: {:.2f}".format(sharpe_ratio))

except Exception as e:
    st.error("Error occurred during optimization: {}".format(str(e)))
