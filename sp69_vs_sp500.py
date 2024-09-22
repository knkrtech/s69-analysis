import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import plotly.graph_objects as go
import requests
from io import StringIO

# Download and process S&P 500 companies data
@st.cache_data
def get_sp500_companies():
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
    response = requests.get(url)
    df = pd.read_csv(StringIO(response.text))
    
    print("Columns in the DataFrame:")
    print(df.columns)
    
    # Ensure 'Symbol' column exists
    if 'Symbol' not in df.columns:
        raise ValueError("'Symbol' column not found in the CSV file")
    
    # Convert symbols to strings and filter out any non-string values
    df['Symbol'] = df['Symbol'].astype(str)
    df = df[df['Symbol'].apply(lambda x: x.isalpha())]  # Keep only alphabetic symbols
    
    # Fetch market cap data using yfinance
    tickers = yf.Tickers(df['Symbol'].tolist())
    market_caps = {}
    for ticker, info in tickers.tickers.items():
        try:
            market_cap = info.info['marketCap']
            if market_cap is not None:
                market_caps[ticker] = market_cap
        except:
            pass
    
    # Add market cap to dataframe and sort
    df['Market Cap'] = df['Symbol'].map(market_caps)
    df = df.sort_values('Market Cap', ascending=False).dropna(subset=['Market Cap']).reset_index(drop=True)
    
    tickers = df.head(69)['Symbol'].tolist()
    
    print("S&P69 Tickers:")
    print(tickers)
    
    return tickers

# Get the top 69 companies by market cap
sp69_tickers = get_sp500_companies()

# Download Historical Data
start_date = '2014-09-01'
end_date = '2024-09-01'

@st.cache_data
def load_data():
    sp69_data = yf.download(sp69_tickers, start=start_date, end=end_date)['Adj Close']
    sp500_data = yf.download('^GSPC', start=start_date, end=end_date)['Adj Close']
    return sp69_data, sp500_data

sp69_data, sp500_data = load_data()

# Handle Missing Data
sp69_data.dropna(axis=1, inplace=True)

# Calculate Daily Returns
sp69_returns = sp69_data.pct_change().dropna()
sp500_returns = sp500_data.pct_change().dropna()

# Simulate Biannual Rebalancing
def rebalance_portfolio(returns, rebalance_freq='6ME'):
    rebalance_dates = returns.resample(rebalance_freq).first().index
    weights = pd.DataFrame(index=returns.index, columns=returns.columns)

    for i in range(len(rebalance_dates)-1):
        start = rebalance_dates[i]
        end = rebalance_dates[i+1]
        period = returns.loc[start:end]

        weight = np.repeat(1/len(returns.columns), len(returns.columns))
        weights.loc[start] = weight

    weights = weights.ffill()
    portfolio_returns = (returns * weights).sum(axis=1)
    return portfolio_returns

sp69_portfolio_returns = rebalance_portfolio(sp69_returns)

initial_investment = 10000

# S&P69 Portfolio Value Over Time
sp69_cumulative_returns = (1 + sp69_portfolio_returns).cumprod() * initial_investment

# S&P500 Index Value Over Time
sp500_cumulative_returns = (1 + sp500_returns).cumprod() * initial_investment

trading_days = 252  # Average trading days in a year
risk_free_rate = 0.02  # 2% annual risk-free rate

# Annualized Return and Volatility
sp69_annual_return = sp69_portfolio_returns.mean() * trading_days
sp69_annual_volatility = sp69_portfolio_returns.std() * np.sqrt(trading_days)

sp500_annual_return = sp500_returns.mean() * trading_days
sp500_annual_volatility = sp500_returns.std() * np.sqrt(trading_days)

# Sharpe Ratio (with 2% risk-free rate)
sp69_sharpe_ratio = (sp69_annual_return - risk_free_rate) / sp69_annual_volatility
sp500_sharpe_ratio = (sp500_annual_return - risk_free_rate) / sp500_annual_volatility

# Sortino Ratio (with 2% risk-free rate)
def sortino_ratio(returns, risk_free_rate, trading_days):
    excess_returns = returns - (risk_free_rate / trading_days)
    negative_returns = excess_returns[excess_returns < 0]
    downside_deviation = negative_returns.std() * np.sqrt(trading_days)
    expected_return = returns.mean() * trading_days
    return (expected_return - risk_free_rate) / downside_deviation

sp69_sortino_ratio = sortino_ratio(sp69_portfolio_returns, risk_free_rate, trading_days)
sp500_sortino_ratio = sortino_ratio(sp500_returns, risk_free_rate, trading_days)

# Maximum Drawdown
def max_drawdown(cumulative_returns):
    rolling_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    return drawdown.min()

sp69_max_drawdown = max_drawdown(sp69_cumulative_returns)
sp500_max_drawdown = max_drawdown(sp500_cumulative_returns)

st.title('The Power of 69: S&P69 vs S&P500')

st.write("""
## Nice: Concentrated Investing That'll Make Your Portfolio Blush

Welcome to Two Asset Management's groundbreaking analysis of the S&P69, a *nice* and concentrated index of the top 69 companies by market capitalization from the S&P500. Our research demonstrates how focusing on the market's biggest players can potentially yield returns that are... well, let's just say they're more than *nice*.

### Our Methodology (No, Not That Kind of Methodology)

1. **Data Source**: We use real-time data from the S&P500 constituents. It's fresher than the produce section at Whole Foods.
2. **Selection Criteria**: We pick the top 69 companies based on market cap. Size matters, folks.
3. **Rebalancing**: Our index gets a makeover every 6 months. It's like a spa day for your portfolio.
4. **Performance Comparison**: We pit the S&P69 against the S&P500 over a 10-year period. It's the financial equivalent of a cage match.

### Limitations and Considerations (The Fine Print)

- **Survivorship Bias**: We assume the current top 69 were always on top. Kind of like assuming your high school quarterback is still cool.
- **Transaction Costs**: Our model doesn't include costs or taxes. In this world, the only certainties are death and... well, just death.
- **Concentration Risk**: The S&P69 is more concentrated than a college student during finals week. Higher risk, but potentially higher reward.

### Why S&P69 with 6-Month Rebalancing Could Be Financially Viable (And Not Just Because We Like the Number)

1. **Lower Turnover**: We only shake things up twice a year. It's like dating in your 30s - quality over quantity.
2. **Focus on Large Caps**: We're talking about the popular kids of the stock market. They're liquid, but not in a "spilled drink" kind of way.
3. **Potential for Outperformance**: By focusing on market leaders, we aim to capture more upside than a Silicon Valley startup's stock options.
4. **Simplicity**: It's so straightforward, even your cousin who still uses a flip phone could understand it.
5. **Scalability**: This strategy could handle more assets than a celebrity divorce lawyer.

Now, let's dive into the data and see if the S&P69 can make your portfolio say "Nice!"
""")

# Plot cumulative returns
fig = go.Figure()
fig.add_trace(go.Scatter(x=sp69_cumulative_returns.index, y=sp69_cumulative_returns, name='S&P69'))
fig.add_trace(go.Scatter(x=sp500_cumulative_returns.index, y=sp500_cumulative_returns, name='S&P500'))
fig.update_layout(title='Cumulative Returns: $10,000 Investment', xaxis_title='Date', yaxis_title='Portfolio Value ($)')
st.plotly_chart(fig)

# Display performance metrics
st.subheader('Performance Metrics')
metrics_df = pd.DataFrame({
    'Metric': ['Annualized Return', 'Annualized Volatility', 'Sharpe Ratio', 'Sortino Ratio', 'Maximum Drawdown'],
    'S&P69': [sp69_annual_return, sp69_annual_volatility, sp69_sharpe_ratio, sp69_sortino_ratio, sp69_max_drawdown],
    'S&P500': [sp500_annual_return, sp500_annual_volatility, sp500_sharpe_ratio, sp500_sortino_ratio, sp500_max_drawdown]
})

# Format the DataFrame
formatted_df = metrics_df.set_index('Metric')
formatted_df = formatted_df.style.format({
    'S&P69': lambda x: f'{x:.2%}' if x in [sp69_annual_return, sp69_annual_volatility, sp69_max_drawdown] else f'{x:.2f}',
    'S&P500': lambda x: f'{x:.2%}' if x in [sp500_annual_return, sp500_annual_volatility, sp500_max_drawdown] else f'{x:.2f}'
})

st.table(formatted_df)

# Final portfolio values
final_sp69_value = sp69_cumulative_returns.iloc[-1]
final_sp500_value = sp500_cumulative_returns.iloc[-1]

st.subheader('Final Portfolio Values')
st.write(f'S&P69: ${final_sp69_value:.2f}')
st.write(f'S&P500: ${final_sp500_value:.2f}')

outperformance = (final_sp69_value - final_sp500_value) / final_sp500_value
st.write(f'S&P69 outperformed S&P500 by {outperformance:.2%}')

st.write("""
## Conclusion: Is 69 the Magic Number?

Our analysis shows that when it comes to investing, 69 might just be the magic number. The S&P69 strategy, focusing on the nice... I mean, top 69 companies by market cap, has shown impressive results compared to the broader S&P500 index.

Key takeaways:
1. Higher Returns: The S&P69 outperformed the S&P500. It's like comparing a sports car to your dad's minivan.
2. Improved Risk-Adjusted Performance: Better Sharpe and Sortino ratios. It's not just about size, it's how you use it.
3. Concentration Benefits: Focusing on market leaders can potentially lead to enhanced performance. It's the "all-star team" approach to investing.

While past performance doesn't guarantee future results (much like your Tinder date's profile pic), the S&P69 strategy offers an intriguing alternative for investors seeking to potentially outperform the market through a concentrated, large-cap focused approach.

Remember to consider your own risk tolerance and investment goals before making any investment decisions. And maybe consult with a financial advisor who doesn't giggle every time you say "S&P69".

*This groundbreaking research is brought to you by Two Asset Management, where we take your assets very seriously, even when we're being cheeky about it.*
""")

# Add a section for current S&P69 constituents
st.subheader('Current S&P69 Constituents')
constituents_df = pd.DataFrame({'Symbol': sp69_tickers})
constituents_df['Company Name'] = constituents_df['Symbol'].apply(lambda x: yf.Ticker(x).info.get('longName', 'N/A'))
st.table(constituents_df)

st.write("""
*Disclaimer: This analysis is for informational purposes only and does not constitute investment advice. Investing in a concentrated portfolio of stocks carries risks, including potential loss of principal. Always do your own research or consult with a financial advisor before making investment decisions. And remember, if your financial advisor starts giggling when you mention the S&P69, it might be time to find a new advisor.*
""")
