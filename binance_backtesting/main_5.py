#!/usr/bin/env python3
"""
main_advanced.py

This script implements two advanced trading strategies based on factor models,
using insights from advanced R-based lectures.

It includes:

Task 1: Factor Regression–Based Trading (using rolling LS regression between Binance and Upbit)
  - We compute a rolling regression between daily returns of two markets.
  - We define the “mispricing spread” as:
       spread = R_Binance - (alpha + beta * R_Upbit)
  - We then calculate a rolling z–score of the spread and generate trading signals:
       Signal = -1 if z > threshold, +1 if z < -threshold, else 0.
  - The strategy returns are computed as the lagged signal multiplied by the daily return.
  
Task 2: Alpha Factors–Based Trading (using composite alpha computed by PCA)
  - We compute three alpha factors from hourly data:
       alpha_A = sqrt(high * low) - VWAP,
       alpha_B = - ((low - close) * (open**5)) / (((low - high) * (close**5)) + epsilon),
       alpha_C = (close - open) / (max(high - low, 0.0001) + 0.001)
  - The three factors are standardized and combined via PCA (keeping the first component).
  - The trading signal is simply generated as signal = sign(composite_alpha)
    (mimicking the idea that positive composite implies bullish, negative implies bearish).
  - The backtest uses hourly close–price returns.
  
Performance metrics (CAGR, maximum drawdown and Sharpe ratio) are computed and results plotted.

Note:
- The original “naive” main.py used fixed thresholds on momentum; here we incorporate robust 
  factor–model ideas and rolling estimation.
- Adjust file paths, windows, and thresholds as needed.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm
from sklearn.decomposition import PCA

# ------------------------------
# PERFORMANCE METRICS FUNCTIONS
# ------------------------------
def calculate_cagr(equity_curve, periods_per_year=365):
    # Ensure the first value is nonzero
    initial = equity_curve.iloc[0] if equity_curve.iloc[0] != 0 else 1.0
    total_return = equity_curve.iloc[-1] / initial
    n_periods = len(equity_curve)
    # Avoid division by zero if n_periods is 0
    if n_periods == 0:
        return np.nan
    cagr = total_return ** (periods_per_year / n_periods) - 1
    return cagr

def calculate_max_drawdown(equity_curve):
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max
    return drawdown.min()

def calculate_sharpe_ratio(returns, risk_free_rate=0, periods_per_year=365):
    excess_returns = returns - risk_free_rate/periods_per_year
    if returns.std() == 0:
        return np.nan
    sharpe = np.sqrt(periods_per_year) * (excess_returns.mean() / returns.std())
    return sharpe

# ------------------------------
# TASK 1: FACTOR REGRESSION–BASED TRADING
# ------------------------------
def load_market_data(binance_file, upbit_file):
    """
    Load and clean daily market data CSV files.
    Expect columns 'open_time' and 'close'. Convert the Unix timestamp and merge.
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
    
    # Align dates and drop missing entries
    df = pd.concat([df_upbit['Upbit'], df_binance['Binance']], axis=1).dropna()
    return df

def compute_returns(df):
    """
    Compute daily percentage returns from price data.
    """
    returns = df.pct_change().dropna()
    return returns

def rolling_factor_regression(returns, window=30):
    """
    Perform a rolling regression of Binance returns on Upbit returns.
    Returns rolling series of alpha and beta estimates.
    """
    y = returns['Binance']
    X = returns['Upbit']
    X = sm.add_constant(X)
    model = RollingOLS(y, X, window=window)
    rres = model.fit()
    alpha_roll = rres.params['const']
    beta_roll  = rres.params['Upbit']
    return alpha_roll, beta_roll

def generate_spread_signal(returns, alpha_series, beta_series, roll_window=30, z_threshold=1.75):
    """
    Given the daily returns and rolling regression parameters, compute:
      modeled return = alpha + beta * Upbit_return
      spread = Binance_return - modeled return.
    Then compute the rolling mean and std of the spread to determine a rolling z–score.
    Generate trading signals:
       If zscore > z_threshold: signal = -1  (short Binance)
       If zscore < -z_threshold: signal = +1 (long Binance)
       Else: 0.
    """
    aligned = returns.copy()
    aligned = aligned.join(alpha_series.rename("alpha")).join(beta_series.rename("beta"))
    aligned['model'] = aligned['alpha'] + aligned['beta'] * aligned['Upbit']
    aligned['spread'] = aligned['Binance'] - aligned['model']
    
    # Rolling statistics of spread:
    aligned['spread_mean'] = aligned['spread'].rolling(window=roll_window, min_periods=roll_window).mean()
    aligned['spread_std'] = aligned['spread'].rolling(window=roll_window, min_periods=roll_window).std()
    aligned = aligned.dropna()
    aligned['zscore'] = (aligned['spread'] - aligned['spread_mean']) / aligned['spread_std']
    
    # Generate signal based on zscore:
    def signal_func(z):
        if z > z_threshold:
            return -1
        elif z < -z_threshold:
            return 1
        else:
            return 0
    aligned['signal'] = aligned['zscore'].apply(signal_func)
    return aligned

def backtest_market_strategy(aligned_data, z_threshold):
    """
    Backtest the market (Task 1) strategy.
    We assume that aligned_data already comes from percentage returns.
    We use the lagged signal to avoid lookahead bias.
    The strategy return is computed as: signal_{t-1} * Binance_return (where Binance_return is already % change)
    """
    aligned_data['signal_lag'] = aligned_data['signal'].shift(1)
    # IMPORTANT: Binance column in aligned_data is already the daily percentage return.
    aligned_data['strategy_returns'] = aligned_data['signal_lag'] * aligned_data['Binance']
    aligned_data['equity_curve'] = (1 + aligned_data['strategy_returns'].fillna(0)).cumprod()
    
    cagr = calculate_cagr(aligned_data['equity_curve'], periods_per_year=365)
    mdd = calculate_max_drawdown(aligned_data['equity_curve'])
    sharpe = calculate_sharpe_ratio(aligned_data['strategy_returns'], periods_per_year=365)
    
    print("Task 1 - Market Factor Strategy Performance:")
    print(f"CAGR: {cagr:.2%}")
    print(f"Max Drawdown: {mdd:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    
    # Plot Equity Curve and z-score of spread
    plt.figure(figsize=(14,8))
    plt.subplot(2,1,1)
    plt.plot(aligned_data.index, aligned_data['equity_curve'], label="Equity Curve")
    plt.title("Task 1: Equity Curve (Factor Regression–Based Strategy)")
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
    plt.show()
    
    return aligned_data

# ------------------------------
# TASK 2: ALPHA FACTORS–BASED TRADING VIA COMPOSITE SIGNAL
# ------------------------------
def load_hourly_data(detail_file):
    """
    Load detailed hourly CSV data.
    Expected columns: 'open_time', 'open', 'high', 'low', 'close', 'volume'.
    """
    df = pd.read_csv(detail_file)
    df['DateTime'] = pd.to_datetime(df['open_time'], unit='ms')
    df.sort_values('DateTime', inplace=True)
    df.set_index('DateTime', inplace=True)
    df = df.fillna(method='ffill')
    return df

def compute_vwap(df, window=24):
    """
    Compute the rolling VWAP using typical price and volume.
    Typical price = (high + low + close) / 3.
    """
    df['typical'] = (df['high'] + df['low'] + df['close']) / 3
    vwap = (df['typical'] * df['volume']).rolling(window=window, min_periods=1).sum() / \
           df['volume'].rolling(window=window, min_periods=1).sum()
    return vwap

def compute_alpha_factors(df):
    """
    Compute alpha factors as described:
      alpha_A = sqrt(high * low) - VWAP (VWAP over 24-hour window)
      alpha_B = - ((low - close) * (open**5)) / (((low - high) * (close**5)) + epsilon)
      alpha_C = (close - open) / (max(high - low, 0.0001) + 0.001)
    """
    epsilon = 1e-8
    df = df.copy()
    df['VWAP'] = compute_vwap(df, window=24)
    df['alpha_A'] = np.sqrt(df['high'] * df['low']) - df['VWAP']
    
    denom = (df['low'] - df['high']) * (df['close']**5) + epsilon
    df['alpha_B'] = - ( (df['low'] - df['close']) * (df['open']**5) ) / denom
    
    df['alpha_C'] = (df['close'] - df['open']) / (np.maximum(df['high'] - df['low'], 0.0001) + 0.001)
    
    return df[['alpha_A', 'alpha_B', 'alpha_C']]

def combine_alphas_via_pca(alpha_df, n_components=1):
    """
    Combine the alpha factors via PCA.
    Here we standardize globally (using all available data) and extract the first component.
    """
    alphas = alpha_df.dropna()
    standardized = (alphas - alphas.mean()) / alphas.std()
    pca = PCA(n_components=n_components)
    composite = pca.fit_transform(standardized)
    composite_df = pd.DataFrame(composite, index=standardized.index, columns=['composite_alpha'])
    return composite_df

def generate_alpha_signal(composite_df):
    """
    Generate a trading signal from the composite alpha.
    For this version we simply use the sign:
         signal = +1 if composite_alpha > 0, else -1.
    """
    df = composite_df.copy()
    df['signal'] = np.sign(df['composite_alpha'])
    return df

def backtest_alpha_strategy(price_df, signal_df, composite_df):
    """
    Backtest the alpha–based trading strategy on hourly data.
    Compute hourly returns from 'close' price and use lagged signals.
    """
    df = price_df.copy()
    df['return'] = df['close'].pct_change()
    # Align signal with price data and fill forward missing signals:
    signal = signal_df['signal'].reindex(df.index).ffill()
    df['signal'] = signal.shift(1)  # Use previous hour's signal to avoid lookahead bias.
    df['strategy_returns'] = df['signal'] * df['return']
    df['equity_curve'] = (1 + df['strategy_returns'].fillna(0)).cumprod()
    
    # For hourly data, assume ~365*24 periods in a year.
    cagr = calculate_cagr(df['equity_curve'], periods_per_year=365*24)
    mdd = calculate_max_drawdown(df['equity_curve'])
    sharpe = calculate_sharpe_ratio(df['strategy_returns'], periods_per_year=365*24)
    
    print("Task 2 - Composite Alpha Strategy Performance:")
    print(f"CAGR: {cagr:.2%}")
    print(f"Max Drawdown: {mdd:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    
    plt.figure(figsize=(14,8))
    plt.subplot(2,1,1)
    plt.plot(df.index, df['equity_curve'], label="Equity Curve")
    plt.title("Task 2: Equity Curve (Composite Alpha Strategy)")
    plt.xlabel("Date")
    plt.ylabel("Equity Value")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2,1,2)
    # Also plot the composite_alpha (which is already globally standardized)
    plt.plot(composite_df.index, composite_df['composite_alpha'], label="Composite Alpha", color='purple')
    plt.axhline(y=0, color='black', linestyle='--', label="Zero")
    plt.title("Task 2: Composite Alpha Signal")
    plt.xlabel("Date")
    plt.ylabel("Composite Alpha")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    return df

# ------------------------------
# MAIN EXECUTION FUNCTION
# ------------------------------
def main():
    # --------- TASK 1: Market Factor Strategy ---------
    binance_file = 'task_1_data/binance.cleaned.csv'   # Adjust path as needed
    upbit_file   = 'task_1_data/upbit_data.cleaned.csv'  # Adjust path as needed
    
    market_df = load_market_data(binance_file, upbit_file)
    returns = compute_returns(market_df)
    
    roll_window_reg = 30
    alpha_roll, beta_roll = rolling_factor_regression(returns, window=roll_window_reg)
    z_threshold = 1.75  # threshold for spread z-score
    aligned = generate_spread_signal(returns, alpha_roll, beta_roll, roll_window=roll_window_reg, z_threshold=z_threshold)
    market_results = backtest_market_strategy(aligned, z_threshold)
    
    # --------- TASK 2: Composite Alpha Strategy ---------
    detail_file = 'task_2_data/binance_1h.csv'  # Adjust path as needed
    hourly_df = load_hourly_data(detail_file)
    
    alphas_df = compute_alpha_factors(hourly_df)
    composite_df = combine_alphas_via_pca(alphas_df, n_components=1)
    alpha_signals = generate_alpha_signal(composite_df)
    alpha_results = backtest_alpha_strategy(hourly_df, alpha_signals, composite_df)

if __name__ == "__main__":
    main()
