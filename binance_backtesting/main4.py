#!/usr/bin/env python3
"""
Main_advanced.py

This script implements two advanced trading strategies based on factor models,
drawing on insights from R lecture notes:
 
Task 1: Factor Regression Based Trading Across Markets
  - Use a rolling regression of Binance returns on Upbit returns to estimate dynamic α and β.
  - Compute the mispricing spread and its rolling z–score.
  - Generate signals based on thresholds and backtest the strategy.
  
Task 2: Alpha Factors Based Trading with Composite Signal
  - Compute three alpha factors from hourly trading data.
  - Combine them via PCA into one composite alpha factor.
  - Generate trading signals based on the composite alpha’s z–score and backtest.

Previous naive strategies (e.g., main.py) used fixed thresholds without dynamic factor modeling.
This script seeks to be robust by leveraging robust rolling estimation and modern Python libraries.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm
from sklearn.decomposition import PCA

# NOTICE: comment this two lines if you do not have scienceplots installed
import scienceplots
plt.style.use(['science','ieee'])

# For performance metrics:
def calculate_cagr(equity_curve, periods_per_year=365):
    n_periods = len(equity_curve)
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    cagr = total_return ** (periods_per_year / n_periods) - 1
    return cagr

def calculate_max_drawdown(equity_curve):
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max
    return drawdown.min()

def calculate_sharpe_ratio(returns, risk_free_rate=0, periods_per_year=365):
    excess_returns = returns - risk_free_rate/periods_per_year
    sharpe = np.sqrt(periods_per_year) * (excess_returns.mean() / excess_returns.std())
    return sharpe

# ------------------------------
# TASK 1: Factor Regression Based Strategy
# ------------------------------
def load_market_data(binance_file, upbit_file):
    """
    Load and clean daily market data CSV files.
    The CSVs must have a column named 'open_time' and 'close'.
    """
    df_binance = pd.read_csv(binance_file)
    df_binance['Date'] = pd.to_datetime(df_binance['open_time'], unit='ms')
    df_binance.sort_values('Date', inplace=True)
    df_binance.set_index('Date', inplace=True)
    df_binance = df_binance.rename(columns={'close': 'Binance'})
    
    df_upbit = pd.read_csv(upbit_file)
    df_upbit['Date'] = pd.to_datetime(df_upbit['open_time'], unit='ms')
    df_upbit.sort_values('Date', inplace=True)
    df_upbit.set_index('Date', inplace=True)
    df_upbit = df_upbit.rename(columns={'close': 'Upbit'})
    
    # Merge and drop missing dates
    df = pd.concat([df_upbit['Upbit'], df_binance['Binance']], axis=1).dropna()
    return df

def compute_returns(df):
    """
    Compute daily returns from prices.
    """
    returns = df.pct_change().dropna()
    return returns

def rolling_factor_regression(returns, window=30):
    """
    Perform a rolling regression of Binance returns on Upbit returns.
    Returns DataFrames with rolling estimates of alpha and beta.
    We'll add a constant to the regressors.
    """
    y = returns['Binance']
    X = returns['Upbit']
    X = sm.add_constant(X)
    models = RollingOLS(y, X, window=window)
    rres = models.fit()
    # Extract coefficients (alpha = const, beta = coefficient on Upbit)
    alpha_rolling = rres.params['const']
    beta_rolling  = rres.params['Upbit']
    return alpha_rolling, beta_rolling

def generate_spread_signal(returns, alpha_series, beta_series, roll_window=30, z_threshold=1.75):
    """
    Compute the mispricing spread based on the rolling regression parameters:
      spread = Binance_return - (alpha + beta * Upbit_return)
    Then compute the rolling zscore of the spread.
    Generate trading signals:
      - if zscore > threshold: signal = -1 (indicating overvaluation; short Binance)
      - if zscore < -threshold: signal = +1 (indicating undervaluation; long Binance)
      - else: signal = 0.
    """
    # Align series
    aligned = returns.join(alpha_series.rename("alpha")).join(beta_series.rename("beta"))
    # Compute modeled Binance return from Upbit:
    aligned['model'] = aligned['alpha'] + aligned['beta']*aligned['Upbit']
    aligned['spread'] = aligned['Binance'] - aligned['model']
    
    # Compute rolling mean and std of spread to get zscore
    aligned['spread_mean'] = aligned['spread'].rolling(window=roll_window, min_periods=roll_window).mean()
    aligned['spread_std'] = aligned['spread'].rolling(window=roll_window, min_periods=roll_window).std()
    aligned = aligned.dropna()
    aligned['zscore'] = (aligned['spread'] - aligned['spread_mean']) / aligned['spread_std']
    
    # Create signal: mean reversion => if z-score is high, expect reversion
    def signal_func(z):
        if z > z_threshold:
            return -1
        elif z < -z_threshold:
            return +1
        else:
            return 0
    aligned['signal'] = aligned['zscore'].apply(signal_func)
    return aligned

def backtest_market_strategy(aligned_data, z_threshold):
    """
    Backtest the strategy. We use the signal from the previous day (lagging to avoid lookahead)
    and assume trade returns equal signal * Binance's return.
    """
    aligned_data['signal_lag'] = aligned_data['signal'].shift(1)
    aligned_data['strategy_returns'] = aligned_data['signal_lag'] * aligned_data['Binance'].pct_change()
    aligned_data['equity_curve'] = (1 + aligned_data['strategy_returns'].fillna(0)).cumprod()
    
    # Calculate performance metrics
    cagr = calculate_cagr(aligned_data['equity_curve'])
    mdd = calculate_max_drawdown(aligned_data['equity_curve'])
    sharpe = calculate_sharpe_ratio(aligned_data['strategy_returns'])
    
    print("Task 1 - Market Factor Strategy Performance:")
    print("CAGR: {:.2%}".format(cagr))
    print("Max Drawdown: {:.2%}".format(mdd))
    print("Sharpe Ratio: {:.2f}".format(sharpe))
    
    # Plot equity curve and z-score
    plt.figure(figsize=(14,6))
    plt.subplot(2,1,1)
    plt.plot(aligned_data.index, aligned_data['equity_curve'], label="Equity Curve")
    plt.title("Task 1: Equity Curve of Factor Regression Strategy")
    plt.xlabel("Date")
    plt.ylabel("Equity Value")
    plt.legend()
    plt.grid(True)
    plt.subplot(2,1,2)
    plt.plot(aligned_data.index, aligned_data['zscore'], label="Spread Z-Score", color='orange')
    plt.axhline(y=z_threshold, color='red', linestyle='--', label="Upper Threshold")
    plt.axhline(y=-z_threshold, color='green', linestyle='--', label="Lower Threshold")
    plt.title("Task 1: Rolling Z-Score of Mispricing Spread")
    plt.xlabel("Date")
    plt.ylabel("Z-Score")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    # plt.show()
    plt.savefig("task2.pdf",dpi=600)
    return aligned_data

# ------------------------------
# TASK 2: Alpha Factors Based Trading with Composite Signal
# ------------------------------
def load_hourly_data(detail_file):
    """
    Load detailed hourly CSV data. The CSV should contain columns: 'open_time', 'open', 'high', 'low', 'close', and 'volume'.
    """
    df = pd.read_csv(detail_file)
    df['DateTime'] = pd.to_datetime(df['open_time'], unit='ms')
    df.sort_values('DateTime', inplace=True)
    df.set_index('DateTime', inplace=True)
    df = df.fillna(method='ffill')
    return df

def compute_vwap(df, window=24):
    """
    Compute a rolling VWAP using the typical price and volume.
    Typical price = (high + low + close)/3
    """
    df['typical'] = (df['high'] + df['low'] + df['close']) / 3
    rolling_vwap = (df['typical'] * df['volume']).rolling(window=window, min_periods=1).sum() / \
                   df['volume'].rolling(window=window, min_periods=1).sum()
    return rolling_vwap

def compute_alpha_factors(df):
    """
    Compute three alpha factors based on the notes:
      alpha_A = sqrt(high * low) - VWAP  (VWAP computed over 24 period rolling window)
      alpha_B = - ((low - close)*(open**5)) / ( ((low - high) * (close**5)) + epsilon )
      alpha_C = (close - open) / (max(high - low, 0.0001) + 0.001)
    """
    epsilon = 1e-8
    df = df.copy()
    # Calculate rolling VWAP over 24 hours:
    df['VWAP'] = compute_vwap(df, window=24)
    df['alpha_A'] = np.sqrt(df['high'] * df['low']) - df['VWAP']
    
    # Avoid division by zero:
    denom = (df['low'] - df['high']) * (df['close']**5) + epsilon
    df['alpha_B'] = - ( (df['low'] - df['close']) * (df['open']**5) ) / denom
    
    # Avoid too small denominator:
    df['alpha_C'] = (df['close'] - df['open']) / (np.maximum(df['high'] - df['low'], 0.0001) + 0.001)
    
    return df[['alpha_A', 'alpha_B', 'alpha_C']]

def combine_alphas_via_pca(alpha_df, n_components=1, roll_window=24):
    """
    Combine the computed alpha factors over time.
    We first standardize the factors, then use PCA over a rolling window (if desired).
    For simplicity here, we perform PCA over the entire sample.
    In production you might use a rolling PCA.
    """
    alphas = alpha_df.dropna()
    # Standardize factors:
    standardized = (alphas - alphas.mean()) / alphas.std()
    pca = PCA(n_components=n_components)
    composite = pca.fit_transform(standardized)
    # Return as DataFrame with the same index:
    composite_df = pd.DataFrame(composite, index=standardized.index, columns=['composite_alpha'])
    return composite_df

def generate_alpha_signal(composite_df, roll_window=24, z_threshold=1.0):
    """
    Compute the rolling z-score of the composite alpha factor.
    Generate signal:
      - if z > threshold: signal = +1 (long)
      - if z < -threshold: signal = -1 (short)
      - else: 0.
    """
    df = composite_df.copy()
    df['mean'] = df['composite_alpha'].rolling(window=roll_window, min_periods=roll_window).mean()
    df['std'] = df['composite_alpha'].rolling(window=roll_window, min_periods=roll_window).std()
    df = df.dropna()
    df['zscore'] = (df['composite_alpha'] - df['mean']) / df['std']
    
    def signal_func(z):
        if z > z_threshold:
            return 1
        elif z < -z_threshold:
            return -1
        else:
            return 0
    df['signal'] = df['zscore'].apply(signal_func)
    return df

def backtest_alpha_strategy(price_df, signal_df):
    """
    Backtest the alpha-based strategy on a chosen asset.
    For demonstration, we compute returns on the 'close' price.
    We use a lagged signal (shifted by 1 period) to avoid lookahead bias.
    """
    df = price_df.copy()
    df['return'] = df['close'].pct_change()
    # Merge with signal by index intersection
    signal = signal_df['signal'].reindex(df.index).ffill()
    df['signal'] = signal.shift(1)  # lag signal
    df['strategy_returns'] = df['signal'] * df['return']
    df['equity_curve'] = (1 + df['strategy_returns'].fillna(0)).cumprod()
    
    cagr = calculate_cagr(df['equity_curve'], periods_per_year=365*24)  # assume hourly data count per year ~365*24
    mdd = calculate_max_drawdown(df['equity_curve'])
    sharpe = calculate_sharpe_ratio(df['strategy_returns'], periods_per_year=365*24)
    
    print("Task 2 - Alpha Factors Composite Strategy Performance:")
    print("CAGR: {:.2%}".format(cagr))
    print("Max Drawdown: {:.2%}".format(mdd))
    print("Sharpe Ratio: {:.2f}".format(sharpe))
    
    # Plot equity curve and composite signal z-score
    plt.figure(figsize=(14,6))
    plt.subplot(2,1,1)
    plt.plot(df.index, df['equity_curve'], label="Equity Curve")
    plt.title("Task 2: Equity Curve of Composite Alpha Strategy")
    plt.xlabel("Date")
    plt.ylabel("Equity Value")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2,1,2)
    plt.plot(signal_df.index, signal_df['zscore'], label="Composite Alpha Z-Score", color='purple')
    plt.axhline(y=1.0, color='red', linestyle='--', label="Upper Threshold")
    plt.axhline(y=-1.0, color='green', linestyle='--', label="Lower Threshold")
    plt.title("Task 2: Rolling Z-Score of Composite Alpha")
    plt.xlabel("Date")
    plt.ylabel("Z-Score")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    # plt.show()
    
    plt.savefig("task1.pdf",dpi=600)
    
    return df

# ------------------------------
# Main Execution
# ------------------------------
def main():
    # --- Task 1 ---
    # Please adjust file names as needed.
    binance_file = 'task_1_data/binance.cleaned.csv'
    upbit_file   = 'task_1_data/upbit_data.cleaned.csv'
    
    market_df = load_market_data(binance_file, upbit_file)
    returns = compute_returns(market_df)
    # Rolling regression with window = 30 days (you can adjust)
    roll_window_reg = 30
    alpha_roll, beta_roll = rolling_factor_regression(returns, window=roll_window_reg)
    
    # Generate mispricing spread signal using same window for z-score
    z_threshold = 1.75  # can tune this threshold
    aligned = generate_spread_signal(returns, alpha_roll, beta_roll, roll_window=roll_window_reg, z_threshold=z_threshold)
    
    # Backtest Task 1 strategy and show metrics and plots
    market_results = backtest_market_strategy(aligned, z_threshold)
    
    # --- Task 2 ---
    # Please adjust file names as needed.
    detail_file = 'task_2_data/binance_1h.csv'
    hourly_df = load_hourly_data(detail_file)
    
    # Compute alpha factors (using detailed price and volume data)
    alphas_df = compute_alpha_factors(hourly_df)
    # Combine alphas using PCA to get one composite factor
    composite_alpha = combine_alphas_via_pca(alphas_df, n_components=1)
    # Generate trading signal from composite factor (using a rolling window of 24 hours)
    alpha_signals = generate_alpha_signal(composite_alpha, roll_window=24, z_threshold=1.0)
    
    # Backtest the composite alpha strategy on hourly 'close' prices.
    alpha_results = backtest_alpha_strategy(hourly_df, alpha_signals)
    
if __name__ == "__main__":
    main()
