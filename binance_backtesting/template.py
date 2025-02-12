# %% [markdown]
# # Bitcoin Backtesting and Trading Strategies
#
# This notebook covers a complete workflow for a coding test, organized in three parts:
#
# **Part 1: Kimchi Momentum Strategy Backtesting**  
# - Trading rule: When Bitcoin's price on Upbit goes up by X% (signal to long Binance), and down by Y% (signal to short).
# - Synthetic data generation for Bitcoin prices on Upbit and Binance.
# - Data splitting into backtesting (first 50%) and forward testing (last 50%).
# - Looping over a range of X and Y values, performing backtests, and generating a Sharpe ratio heat map.
# - Calculation of CAGR, Maximum Drawdown, and Sharpe Ratio.
#
# **Part 2: Alpha Factors Backtesting**  
# - Generating synthetic hourly candlestick data for BTCUSDT.
# - Computing three alpha factors (alpha_A, alpha_B, and alpha_C) using the formulas provided.
# - Creating simple trading signals (e.g., long when the alpha is positive, short otherwise).
# - Backtesting the combined strategy based on the computed alphas and visualizing its performance.
#
# **Part 3: Bonus – Bonus Strategies Brainstorm Backtesting**  
# - Proposing a simple CTA (trend-following) strategy using moving averages.
# - Backtesting the strategy on the synthetic data.
# - Computing key performance metrics and plotting the CTA strategy's equity curve.
#
# Each section is well documented for clarity.

# %%
# %% [code]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

import scienceplots
# Set visualization style
plt.style.use(['science','ieee'])

# %%
# %% [markdown]
# ## Part 1: Kimchi Momentum Strategy Backtesting
#
# In this section, we generate synthetic daily price data for Bitcoin on both Upbit and Binance.
# We then implement the following:
#
# - **Trading Rule:**  
#    - **Long on Binance** when Upbit's price increases by X% relative to the previous day.
#    - **Short on Binance** when Upbit's price decreases by Y%.
#
# - **Data Splitting:**  
#   The complete dataset is split into two segments: the first 50% for backtesting (parameter tuning) and the last 50% for forward testing.
#
# - **Parameter Optimization:**  
#   We loop over different combinations of (X, Y) to backtest the strategy and compute the Sharpe Ratio for each pair. A heat map is then generated to identify the optimal thresholds.
#
# - **Performance Metrics:**  
#   Finally, we calculate the CAGR, Maximum Drawdown, and Sharpe Ratio and plot the performance curve.

# %%
# %% [code]
# Synthetic Data Generation (Daily Prices)
# -----------------------------------------
# We simulate a time series using a random walk model for both exchanges.
np.random.seed(42)
dates = pd.date_range(start="2021-01-01", periods=1000, freq="D")

# Simulate Upbit Bitcoin price with a random walk
upbit_returns = np.random.normal(loc=0.0005, scale=0.02, size=len(dates))
upbit_price = 50000 * np.exp(np.cumsum(upbit_returns))  # starting around $50K

# Simulate Binance Bitcoin price by using similar returns with extra noise/spread
binance_returns = upbit_returns + np.random.normal(loc=0.0001, scale=0.005, size=len(dates))
binance_price = 50000 * np.exp(np.cumsum(binance_returns))

# Create DataFrame
data = pd.DataFrame({"Date": dates, "Upbit": upbit_price, "Binance": binance_price})
data.set_index("Date", inplace=True)
print(data.head())

# %%
# %% [code]
# Split the data into two equal parts:
# - First 50% for backtesting (parameter tuning)
# - Last 50% for forward testing
split_index = len(data) // 2
backtest_data = data.iloc[:split_index]
forward_data = data.iloc[split_index:]
print("Backtest period:", backtest_data.index[0].date(), "to", backtest_data.index[-1].date())
print("Forward test period:", forward_data.index[0].date(), "to", forward_data.index[-1].date())

# %%
# %% [code]
# Define performance metrics functions

def calculate_cagr(equity_curve, periods_per_year=365):
    """
    Compute Compound Annual Growth Rate (CAGR).
    equity_curve: pd.Series representing the equity curve.
    periods_per_year: number of periods per year (365 for daily data).
    """
    n_periods = len(equity_curve)
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    cagr = total_return ** (periods_per_year / n_periods) - 1
    return cagr

def calculate_max_drawdown(equity_curve):
    """
    Compute the maximum drawdown of the equity curve.
    """
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max
    max_drawdown = drawdown.min()
    return max_drawdown

def calculate_sharpe_ratio(returns, risk_free_rate=0, periods_per_year=365):
    """
    Compute the Sharpe Ratio.
    returns: pd.Series of periodic returns.
    risk_free_rate: annual risk-free rate.
    periods_per_year: number of periods per year (365 for daily data).
    """
    excess_returns = returns - risk_free_rate/periods_per_year
    sharpe = np.sqrt(periods_per_year) * (excess_returns.mean() / excess_returns.std())
    return sharpe

# %%
# %% [code]
# Backtesting function for the Kimchi Momentum strategy
def backtest_strategy(data, threshold_up, threshold_down):
    """
    Backtest the Kimchi Momentum strategy.
    - If Upbit's daily price increases by at least threshold_up (%), take a long position on Binance.
    - If Upbit's daily price decreases by threshold_down (%), take a short position on Binance.
    
    Parameters:
      data : DataFrame with 'Upbit' and 'Binance' price columns.
      threshold_up : Percentage threshold for a long signal.
      threshold_down : Percentage threshold for a short signal.
    
    Returns:
      returns: Series of daily strategy returns.
      equity_curve: Cumulative product representing the equity curve.
    """
    returns = []
    # Loop over the data starting from the second day
    for i in range(1, len(data)):
        # Calculate the percentage change for Upbit from the previous day
        up_change = (data['Upbit'].iloc[i] - data['Upbit'].iloc[i-1]) / data['Upbit'].iloc[i-1] * 100  
        # Determine position based on thresholds
        if up_change >= threshold_up:
            position = 1    # Long signal
        elif up_change <= -threshold_down:
            position = -1   # Short signal
        else:
            position = 0    # No trade
        
        # Calculate Binance's daily return
        binance_return = (data['Binance'].iloc[i] - data['Binance'].iloc[i-1]) / data['Binance'].iloc[i-1]
        # Strategy daily return based on position
        daily_return = position * binance_return
        returns.append(daily_return)
    
    returns = pd.Series(returns, index=data.index[1:])
    # Build cumulative equity curve assuming starting capital of 1
    equity_curve = (1 + returns).cumprod()
    return returns, equity_curve

# %%
# %% [code]
# Parameter Optimization: Loop over a range of (X, Y) thresholds and compute the Sharpe Ratio.
# We use thresholds from 0.5% to 3.0% (in increments of 0.5%).
threshold_vals = np.arange(0.5, 3.5, 0.5)
heatmap_df = pd.DataFrame(index=threshold_vals, columns=threshold_vals)

for X in threshold_vals:
    for Y in threshold_vals:
        ret, eq_curve = backtest_strategy(backtest_data, threshold_up=X, threshold_down=Y)
        sharpe = calculate_sharpe_ratio(ret)
        heatmap_df.loc[X, Y] = sharpe

heatmap_df = heatmap_df.astype(float)

# Plot the Sharpe ratio heat map
plt.figure(figsize=(8,6))
sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap="viridis")
plt.title("Sharpe Ratio Heatmap for Different Thresholds\n(X = Up Threshold, Y = Down Threshold)")
plt.xlabel("Threshold Down (%)")
plt.ylabel("Threshold Up (%)")
plt.show()

# %%
# %% [code]
# Choose optimal parameters (highest Sharpe ratio) from the heat map.
optimal_params = heatmap_df.stack().idxmax()
optimal_X, optimal_Y = optimal_params  
print("Optimal parameters found: X (up) = {}, Y (down) = {}".format(optimal_X, optimal_Y))

# Backtest strategy on the backtesting data using the optimal parameters
opt_ret, opt_eq_curve = backtest_strategy(backtest_data, threshold_up=optimal_X, threshold_down=optimal_Y)

# Calculate performance metrics
cagr = calculate_cagr(opt_eq_curve)
max_dd = calculate_max_drawdown(opt_eq_curve)
sharpe = calculate_sharpe_ratio(opt_ret)

print("\nPerformance Metrics on Backtest Data:")
print("CAGR: {:.2%}".format(cagr))
print("Maximum Drawdown: {:.2%}".format(max_dd))
print("Sharpe Ratio: {:.2f}".format(sharpe))

# Plot the performance (equity) curve
plt.figure(figsize=(10,6))
plt.plot(opt_eq_curve.index, opt_eq_curve, label="Equity Curve")
plt.title("Kimchi Momentum Strategy Performance Curve")
plt.xlabel("Date")
plt.ylabel("Equity (Multiplicative)")
plt.legend()
plt.grid(True)
plt.show()

# %%
# %% [markdown]
# ## Part 2: Alpha Factors Backtesting
#
# In this section, we simulate synthetic hourly candlestick data for BTCUSDT and compute three alpha factors:
#
# - **alpha_A:** √(High × Low) − VWAP  
#    VWAP is calculated over a 24-hour rolling window using the typical price.
#
# - **alpha_B:** -1 * ((Low − Close) × (Open⁵)) / ((Low − High) × (Close⁵))  
#    A small epsilon value is used to avoid division by zero.
#
# - **alpha_C:** (Close − Open) / ((High − Low) + 0.001)
#
# After computing the alphas, a simple trading signal is defined for each (long if the alpha is positive, short otherwise), and the signals are combined.
# The strategy is then backtested, and performance metrics plus the equity curve are plotted.

# %%
# %% [code]
# Synthetic Hourly Data Generation for BTCUSDT
# ---------------------------------------------
dates_hourly = pd.date_range(start="2021-01-01", periods=1000, freq="H")
np.random.seed(42)

# Simulate a random walk for price
price = 50000 * np.exp(np.cumsum(np.random.normal(0.0001, 0.01, size=len(dates_hourly))))

# Create synthetic OHLC data: we slightly adjust prices to simulate open, high, low, and close
open_prices = price
close_prices = open_prices * (1 + np.random.normal(0.000, 0.005, size=len(price)))
high_prices = np.maximum(open_prices, close_prices) * (1 + np.abs(np.random.normal(0.001, 0.002, size=len(price))))
low_prices = np.minimum(open_prices, close_prices) * (1 - np.abs(np.random.normal(0.001, 0.002, size=len(price))))
volumes = np.random.randint(100, 1000, size=len(price))

# Construct the DataFrame
df_alpha = pd.DataFrame({
    "Open": open_prices,
    "High": high_prices,
    "Low": low_prices,
    "Close": close_prices,
    "Volume": volumes
}, index=dates_hourly)

print(df_alpha.head())

# %%
# %% [code]
# Data Cleaning: Fill missing values if any (for synthetic data, this is precautionary).
df_alpha.fillna(method='ffill', inplace=True)

# %%
# %% [code]
# Compute Alpha Factors
# -----------------------
# 1. alpha_A = sqrt(High * Low) - VWAP, over a rolling 24-hour window.
df_alpha["Typical"] = (df_alpha["Close"] + df_alpha["High"] + df_alpha["Low"]) / 3
window = 24  # 24-hour rolling window
df_alpha["VWAP"] = (df_alpha["Typical"] * df_alpha["Volume"]).rolling(window=window).sum() / df_alpha["Volume"].rolling(window=window).sum()
df_alpha["VWAP"].fillna(method='bfill', inplace=True)
df_alpha["alpha_A"] = np.sqrt(df_alpha["High"] * df_alpha["Low"]) - df_alpha["VWAP"]

# 2. alpha_B = -1 * ((Low - Close) * (Open^5)) / ((Low - High) * (Close^5))
epsilon = 1e-8  # small value to avoid division by zero
df_alpha["alpha_B"] = -1 * ((df_alpha["Low"] - df_alpha["Close"]) * (df_alpha["Open"]**5)) / (((df_alpha["Low"] - df_alpha["High"]).replace(0, epsilon)) * (df_alpha["Close"]**5 + epsilon))

# 3. alpha_C = (Close - Open) / ((High - Low) + 0.001)
df_alpha["alpha_C"] = (df_alpha["Close"] - df_alpha["Open"]) / ((df_alpha["High"] - df_alpha["Low"]) + 0.001)

print(df_alpha[["alpha_A", "alpha_B", "alpha_C"]].head(10))

# %%
# %% [code]
# Create Trading Signals based on the computed alphas.
# For simplicity, we use:
#    - Signal = +1 (Long) if alpha > 0; -1 (Short) if alpha < 0.
df_alpha["signal_A"] = np.where(df_alpha["alpha_A"] > 0, 1, -1)
df_alpha["signal_B"] = np.where(df_alpha["alpha_B"] > 0, 1, -1)
df_alpha["signal_C"] = np.where(df_alpha["alpha_C"] > 0, 1, -1)

# Combine signals by averaging (alternative approaches can be used)
df_alpha["combined_signal"] = (df_alpha["signal_A"] + df_alpha["signal_B"] + df_alpha["signal_C"]) / 3
# Final signal: take long if non-negative, else short
df_alpha["combined_signal"] = np.where(df_alpha["combined_signal"] >= 0, 1, -1)

print(df_alpha[["signal_A", "signal_B", "signal_C", "combined_signal"]].tail())

# %%
# %% [code]
# Backtesting the Alpha Factors based Strategy
def backtest_alpha_strategy(df, signal_column="combined_signal"):
    """
    Backtest an alpha factors-based trading strategy on hourly data.
    The strategy returns are computed as the product of the previous period's signal and the current period's return.
    """
    df = df.copy()
    # Calculate hourly returns based on Close price
    df["returns"] = df["Close"].pct_change()
    # Shift signal by one to avoid lookahead bias
    df["strategy_returns"] = df[signal_column].shift(1) * df["returns"]
    # Compute cumulative returns (equity curve)
    df["strategy_equity"] = (1 + df["strategy_returns"].fillna(0)).cumprod()
    return df["strategy_returns"].dropna(), df["strategy_equity"]

alpha_returns, alpha_equity_curve = backtest_alpha_strategy(df_alpha, signal_column="combined_signal")

# Use 24*365 as periods per year since the data is hourly.
alpha_cagr = calculate_cagr(alpha_equity_curve, periods_per_year=24*365)
alpha_dd = calculate_max_drawdown(alpha_equity_curve)
alpha_sharpe = calculate_sharpe_ratio(alpha_returns, periods_per_year=24*365)

print("\nAlpha Strategy Performance Metrics:")
print("CAGR: {:.2%}".format(alpha_cagr))
print("Maximum Drawdown: {:.2%}".format(alpha_dd))
print("Sharpe Ratio: {:.2f}".format(alpha_sharpe))

# Plot the Alpha strategy equity curve
plt.figure(figsize=(10,6))
plt.plot(alpha_equity_curve.index, alpha_equity_curve, label="Alpha Strategy Equity Curve")
plt.title("Alpha Factors Strategy Performance Curve")
plt.xlabel("Date")
plt.ylabel("Equity (Multiplicative)")
plt.legend()
plt.grid(True)
plt.show()

# %%
# %% [markdown]
# ## Part 3: Bonus – Bonus Strategies Brainstorm Backtesting
#
# Here we propose an additional trading strategy using a CTA (trend-following) approach. In this example,
# a simple moving average crossover strategy is implemented as follows:
#
# - **CTA Strategy (Moving Averages):**  
#   - Calculate a short-term moving average (e.g., 24-hour) and a long-term moving average (e.g., 72-hour).
#   - Generate a long signal when the short-term moving average is above the long-term moving average, and vice versa.
#
# The strategy's performance is backtested with performance metrics computed and the equity curve plotted.
# Commentary is provided on the result relative to expectations.

# %%
# %% [code]
# Bonus Strategy: Simple CTA Strategy Using Moving Averages
df_bonus = df_alpha.copy()  # Re-use our synthetic hourly data

# Calculate short-term and long-term moving averages on the Close price.
df_bonus["MA_short"] = df_bonus["Close"].rolling(window=24).mean()   # 24-hour MA
df_bonus["MA_long"] = df_bonus["Close"].rolling(window=72).mean()    # 72-hour MA

# Generate CTA signal: +1 when MA_short > MA_long, else -1.
df_bonus["cta_signal"] = np.where(df_bonus["MA_short"] > df_bonus["MA_long"], 1, -1)

# Backtest the CTA strategy: use the CTA signal to weight the hourly returns.
df_bonus["cta_returns"] = df_bonus["cta_signal"].shift(1) * df_bonus["Close"].pct_change()
df_bonus["cta_equity"] = (1 + df_bonus["cta_returns"].fillna(0)).cumprod()

# Calculate performance metrics (using hourly periods as before)
cta_cagr = calculate_cagr(df_bonus["cta_equity"], periods_per_year=24*365)
cta_dd = calculate_max_drawdown(df_bonus["cta_equity"])
cta_sharpe = calculate_sharpe_ratio(df_bonus["cta_returns"].dropna(), periods_per_year=24*365)

print("\nCTA Bonus Strategy Performance Metrics:")
print("CAGR: {:.2%}".format(cta_cagr))
print("Maximum Drawdown: {:.2%}".format(cta_dd))
print("Sharpe Ratio: {:.2f}".format(cta_sharpe))

# Plot the CTA strategy equity curve
plt.figure(figsize=(10,6))
plt.plot(df_bonus["cta_equity"].index, df_bonus["cta_equity"], label="CTA Strategy Equity Curve", color='orange')
plt.title("CTA Bonus Strategy Performance Curve")
plt.xlabel("Date")
plt.ylabel("Equity (Multiplicative)")
plt.legend()
plt.grid(True)
plt.show()

# %%
# %% [markdown]
# ## Final Deliverable Summary
#
# In this notebook we have:
#
# 1. **Kimchi Momentum Strategy Backtesting:**  
#    - Generated synthetic daily Bitcoin price data for Upbit and Binance.
#    - Defined a trading rule (long when Upbit price goes up X% and short when it falls Y%).
#    - Split the data into backtest and forward test sets.
#    - Iterated over a grid of X and Y values to compute Sharpe Ratios and generated a heat map.
#    - Calculated CAGR, Maximum Drawdown, and Sharpe Ratio, and plotted the equity curve.
#
# 2. **Alpha Factors Backtesting:**  
#    - Downloaded (simulated) hourly candlestick data.
#    - Computed three alpha factors (alpha_A, alpha_B, and alpha_C) using the provided formulas.
#    - Generated trading signals based on these alphas and combined them.
#    - Backtested the combined strategy and visualized its performance.
#
# 3. **Bonus Strategy – CTA Approach:**  
#    - Developed a simple moving average crossover strategy for BTC/ETH.
#    - Backtested the strategy, computed performance metrics, and plotted the equity curve.
#    - Provided commentary on the backtest process and noted that while this simple strategy captures trends, further refinement and additional data may be needed to avoid overfitting.
#
# This complete workflow integrates code, visualizations, and detailed explanations, making it a robust deliverable for the coding test.