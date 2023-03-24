import streamlit as st
import yfinance as yf
from datetime import date
from pypfopt import expected_returns, risk_models, EfficientFrontier
import plotly.graph_objs as go


# Set app title
st.title("Optimal Portfolio Allocation")


# Create search bar
st.sidebar.subheader("Search for a company")
symbol = st.sidebar.text_input("Ticker symbol (e.g. AAPL)")


# Load data for selected company
today = date.today().strftime('%Y-%m-%d')
if symbol:
    stock = yf.Ticker(symbol)
    df = stock.history(period="max")[['Close']]
    df = df.rename(columns={'Close': symbol})


# Create input fields for user
st.sidebar.subheader("Portfolio Parameters")
initial_investment = st.sidebar.number_input("Initial investment amount ($)", value=100000, step=1000)
target_return = st.sidebar.number_input("Target annualized return (%)", value=10.0, step=0.5)
risk = st.sidebar.number_input("Desired annualized volatility (%)", value=20.0, step=0.5)


# Optimize portfolio
if symbol:
    # Calculate expected returns and covariance matrix
    mu = expected_returns.mean_historical_return(df)
    S = risk_models.sample_cov(df)

    ef = EfficientFrontier(mu, S)
    weights = ef.efficient_return(target_return/100, market_neutral=True, weight_bounds=(0,1))

    # Calculate allocation of shares
    latest_prices = stock.history(period="1d")['Close'].values[0]
    allocation = {s: round(weights[s] * initial_investment / latest_prices, 2) for s in weights.keys()}

    # Calculate portfolio statistics
    portfolio_value = sum([allocation[s] * latest_prices[i] for i, s in enumerate(allocation.keys())])
    returns = (df.pct_change().mean() * allocation).sum()
    annualized_return = (1 + returns) ** 252 - 1
    cov_matrix = df.pct_change().cov()
    annualized_volatility = (cov_matrix.mul(allocation, axis=0).mul(allocation, axis=1).sum().sum() * 252) ** 0.5
    sharpe_ratio = (annualized_return - 0.02) / annualized_volatility

    # Visualize portfolio performance
    portfolio_returns = (df.pct_change() * allocation).sum(axis=1)
    cumulative_returns = (1 + portfolio_returns).cumprod()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns.values, name='Portfolio'))
    fig.add_trace(go.Scatter(x=cumulative_returns.index, y=(1 + df.pct_change().mean(axis=1)).cumprod().values, name='Benchmark'))
    fig.update_layout(title='Portfolio Performance', xaxis_title='Date', yaxis_title='Cumulative Returns')
    st.plotly_chart(fig, use_container_width=True)

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
