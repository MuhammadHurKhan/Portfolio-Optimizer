import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from pypfopt import expected_returns, risk_models, EfficientFrontier, objective_functions


# Define function to get stock data
@st.cache
def get_data():
    # Download stock data from Yahoo Finance
    sp500 = yf.download("^GSPC", start="2010-01-01", end="2022-03-23")
    return sp500


# Define function to optimize portfolio
def optimize_portfolio(data, init_investment, target_return, risk):
    # Calculate expected returns and covariance matrix
    mu = expected_returns.mean_historical_return(data)
    S = risk_models.sample_cov(data)

    # Define optimization objective
    if risk == 'Minimize volatility':
        ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
        ef.min_volatility()
    else:
        ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
        ef.efficient_return(target_return / 100, objective_functions.L2_reg, None, risk)

    # Get optimized portfolio weights
    weights = ef.clean_weights()
    weights = {k: v * init_investment for k, v in weights.items()}
    weights = {k: round(v / data[k][-1], 2) for k, v in weights.items()}

    return weights


# Define app
def app():
    # Set page title
    st.set_page_config(page_title='Portfolio Optimizer')

    # Set page header
    st.header('Portfolio Optimizer')

    # Get stock data
    data = get_data()

    # Show data
    st.subheader('Stock Data')
    st.write(data.tail())

    # Get user inputs
    init_investment = st.number_input('Initial investment amount ($)', value=100000, step=10000)
    target_return = st.number_input('Desired annual return (%)', value=10.0, step=0.1)
    risk = st.selectbox('Risk', ['Minimize volatility', 'Target volatility'])

    # Optimize portfolio
    try:
        weights = optimize_portfolio(data, init_investment, target_return, risk)

        # Show optimized portfolio weights
        st.subheader('Optimized Portfolio Weights')
        st.write(pd.Series(weights).to_frame('No. of Shares'))

    except Exception as e:
        st.error(f'Error occurred during optimization: {str(e)}')


# Run app
if __name__ == '__main__':
    app()
