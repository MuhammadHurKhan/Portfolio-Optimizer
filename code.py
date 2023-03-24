import streamlit as st
import yfinance as yf
from datetime import date
from pypfopt import expected_returns, risk_models, EfficientFrontier
import plotly.graph_objs as go

# Set app title
st.title("Portfolio Performance Comparison")

# Create search bar
st.sidebar.subheader("Add stocks to your portfolio")
symbols = st.sidebar.text_input("Enter ticker symbols separated by commas (e.g. AAPL,MSFT,AMZN)").upper()
symbols_list = [symbol.strip() for symbol in symbols.split(",")]

# Create input fields for user
st.sidebar.subheader("Number of shares for each stock")
shares_list = []
for symbol in symbols_list:
    shares = st.sidebar.number_input(f"{symbol} shares", value=100, step=1)
    shares_list.append(shares)

initial_investment = st.sidebar.number_input("Initial investment amount ($)", value=100000, step=1000)

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
    sp500_annual_return = (sp500_hist['Close'][-1] / sp500_hist['Close'][0]) ** (1/len(sp500_hist)) - 1

    nasdaq = yf.Ticker('^IXIC')
    nasdaq_hist = nasdaq.history(start="2010-01-01", end=today)
    nasdaq_annual_return = (nasdaq_hist['Close'][-1] / nasdaq_hist['Close'][0]) ** (1/len(nasdaq_hist)) - 1

    # Visualize portfolio performance
    portfolio_returns = (df.pct_change() * allocation).sum(axis=1)
    cumulative_returns = (1 + portfolio_returns).cumprod()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cumulative_returns.index, y= (1 + df['SP500'].pct_change().mean(axis=1)).cumprod().values, name='S&P 500'))
    fig.add_trace(go.Scatter(x=cumulative_returns.index, y=(1 + df['NASDAQ'].pct_change().mean(axis=1)).cumprod().values, name='NASDAQ'))
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

