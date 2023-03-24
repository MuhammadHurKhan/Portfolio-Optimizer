import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import date
from pypfopt import expected_returns, risk_models, EfficientFrontier


def get_portfolio_allocation(symbols_list, initial_investment):
    # Load data for selected companies
    today = date.today().strftime('%Y-%m-%d')
    data = yf.download(symbols_list, start="2010-01-01", end=today, group_by='ticker')['Close']
    df = data.dropna()

    # Optimize portfolio
    if len(df.columns) > 0:
        # Calculate expected returns and covariance matrix
        mu = expected_returns.mean_historical_return(df)
        S = risk_models.sample_cov(df)

        ef = EfficientFrontier(mu, S)
        weights = ef.max_sharpe()

        # Calculate allocation of shares
        latest_prices = df.iloc[-1]
        allocation = {s: round(weights[s] * initial_investment / latest_prices[i], 2) for i, s in enumerate(weights.keys())}

        # Calculate portfolio statistics
        portfolio_value = sum([allocation[s] * latest_prices[i] for i, s in enumerate(allocation.keys())])
        returns = (df.pct_change().mean() * allocation).sum()
        annualized_return = (1 + returns) ** 252 - 1
        cov_matrix = df.pct_change().cov()
        annualized_volatility = (cov_matrix.mul(allocation, axis=0).mul(allocation, axis=1).sum().sum() * 252) ** 0.5
        sharpe_ratio = (annualized_return - 0.02) / annualized_volatility

        # Calculate S&P 500 and NASDAQ annual returns
        sp500 = yf.Ticker('^GSPC')
        sp500_hist = sp500.history(start="2010-01-01", end=today)
        sp500_annual_return = (sp500_hist['Close'][-1] / sp500_hist['Close'][0]) ** (1/len(sp500_hist['Close']) - 1)
        nasdaq = yf.Ticker('^IXIC')
        nasdaq_hist = nasdaq.history(start="2010-01-01", end=today)
        nasdaq_annual_return = (nasdaq_hist['Close'][-1] / nasdaq_hist['Close'][0]) ** (1/len(nasdaq_hist['Close'])) - 1

        # Visualize portfolio performance
        portfolio_returns = (df.pct_change() * allocation).sum(axis=1)
        cumulative_returns = (1 + portfolio_returns).cumprod()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cumulative_returns.index, y=(1 + sp500_hist['Close'].pct_change()).cumprod(), name='S&P 500'))
        fig.add_trace(go.Scatter(x=cumulative_returns.index, y=(1 + nasdaq_hist['Close'].pct_change()).cumprod(), name='NASDAQ'))
        fig.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns.values, name='Portfolio'))
        fig.update_layout(title='Portfolio Performance', xaxis_title='Date', yaxis_title='Cumulative Returns')
        st.plotly_chart(fig, use_container_width=True)

        # Show allocation of shares
        st.subheader("Allocation of Shares")
        for s, a in allocation.items():
            st.write(f"{s}: {a} shares")

        # Show portfolio statistics
        st.subheader("Portfolio Statistics")
        st.write("Portfolio Value: $", round(portfolio_value, 2))
        st.write("Portfolio Return: ", round(returns * 100, 2), "%")
        st.write("Annualized Return: ", round(annualized_return * 100, 2), "%")
        st.write("Annualized Volatility: ", round(annualized_volatility * 100, 2), "%")
        st.write("Sharpe Ratio: ", round(sharpe_ratio, 2))
        st.write("S&P 500 Annual Return: ", round(sp500_annual_return * 100, 2), "%")
        st.write("NASDAQ Annual Return: ", round(nasdaq_annual_return * 100, 2), "%")
        # Show allocation of shares
        st.subheader("Allocation of Shares")
        allocation_df = pd.DataFrame(allocation.items(), columns=['Company', 'Shares'])
        st.dataframe(allocation_df)

        # Show portfolio value
        st.subheader("Portfolio Value")
        st.write("$", round(portfolio_value, 2))
    else: 
        st.write("Please enter valid company names and corresponding shares")
    return

    
