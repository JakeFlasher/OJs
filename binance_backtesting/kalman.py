#!/usr/bin/env python3
"""
Enhanced Task 2 Script using Kalman Filter for Dynamic Hedge Ratio Estimation

This script implements an improved pairs trading strategy for BTC/USDT 
using daily data. It:
  - Loads merged data from futures, spot, and liquidity CSVs.
  - Computes three alpha factors (momentum, a naive RSI-based, and liquidity rank)
    and forms a composite alpha.
  - Applies a Kalman filter to estimate the dynamic hedge ratio (γ) and intercept (μ)
    in the regression: log(SpotClose) = γ * log(FutClose) + μ + ε.
  - Computes the spread: spread = log(SpotClose) - γ*log(FutClose) - μ
  - Uses a pairs trading strategy (with entry, exit, and stop-loss rules guided by
    the spread z-score and composite alpha) to generate trading signals.
  - Computes performance metrics and displays three plots:
      1) Spread evolution,
      2) Composite alpha signal, and
      3) Cumulative PnL.
      
References:
  - Kalman filter regression techniques for pairs trading (see e.g. "A Kalman Filter
    Approach to Trading" articles online).
  - Alpha factors are inspired by momentum and naive RSI methods.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 1) Data Loading & Preparation
# -----------------------------
def load_cryptodata(futures_csv, spots_csv, liquidity_csv):
    """
    Load futures, spot, and liquidity data; merge based on Date index.
    
    Assumes:
      - CSV files contain a column 'Date' (in a parseable format).
      - Futures file: e.g. contains columns like 'FutOpen', 'FutHigh', 'FutLow', 'FutClose'
      - Spots file: e.g. contains columns like 'SpotOpen', 'SpotHigh', 'SpotLow', 'SpotClose'
      - Liquidity file: contains a 'Liquidity' column.
    """
    df_fut = pd.read_csv(futures_csv, parse_dates=['file_date'], index_col='file_date')
    df_spot = pd.read_csv(spots_csv, parse_dates=['file_date'], index_col='file_date')
    df_liq = pd.read_csv(liquidity_csv, parse_dates=['file_date'], index_col='file_date')
    df_liq['Liquidity'] = df_liq['original_quantity'] * df_liq['price']

    # Merge the dataframes by date
    df_all = df_spot.join(df_fut, lsuffix='_spot', rsuffix='_fut', how='inner')
    df_all = df_all.join(df_liq, how='inner')
    
    # (Optional) Rename columns if needed:
    # For example, ensure columns: ['SpotOpen','SpotHigh','SpotLow','SpotClose',
    #                               'FutOpen','FutHigh','FutLow','FutClose',
    #                               'Liquidity']  
    return df_all.dropna()

# -----------------------------
# 2) Compute Alpha Signals
# -----------------------------
def compute_alphas(df):
    """
    Compute alpha signals using spot price and liquidity.
      - alpha_mom: 3-day price momentum on SpotClose.
      - alpha_rsi: Naive RSI-based measure scaled from 0 to 1 (oversold when >0).
      - alpha_liq: Liquidity ranking (dense rank normalized to [0,1]).
    The composite alpha is a weighted sum of the three.
    """
    out = df.copy()
    # (A) Momentum alpha: 3-day percentage return on SpotClose.
    out['alpha_mom'] = out['SpotClose'].pct_change(3)
    
    # (B) Naive RSI-based alpha
    window = 14
    delta = out['SpotClose'].diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0.0).rolling(window=window).mean()
    out['rsi'] = 100 - 100 / (1 + gain / (loss + 1e-9))
    # Rescale RSI to create alpha such that values > 0 indicate oversold conditions.
    out['alpha_rsi'] = 1.0 - out['rsi'] / 100.0
    
    # (C) Liquidity alpha: use rank over Liquidity column
    out['alpha_liq'] = out['Liquidity'].rank(method='dense') / len(out)
    
    # Composite alpha (weights can be tuned)
    w_mom = 0.5; w_rsi = 0.3; w_liq = 0.2
    out['alpha_composite'] = (w_mom * out['alpha_mom'].fillna(0) +
                              w_rsi * out['alpha_rsi'].fillna(0) +
                              w_liq * out['alpha_liq'].fillna(0))
    return out

# -----------------------------
# 3) Kalman Filter Estimation
# -----------------------------
def kalman_filter_estimate(df, delta=1e-5, R_var=1e-4):
    """
    Apply a Kalman filter to estimate the dynamic hedge ratio (gamma) and intercept (mu)
    in the regression:
        log(SpotClose) = gamma * log(FutClose) + mu + ε
        
    Parameters:
      - df: DataFrame with 'SpotClose' and 'FutClose'
      - delta: scalar to set the state noise covariance (Q = delta * I)
      - R_var: observation noise variance
      
    Returns:
      - gamma_series: estimated dynamic hedge ratios (pd.Series)
      - mu_series: estimated dynamic intercepts (pd.Series)
    """
    # Convert prices to log scale
    y = np.log(df['SpotClose'] + 1e-9)
    x = np.log(df['FutClose'] + 1e-9)
    n = len(df)
    
    # Initialize state: theta = [gamma, mu]. Start with gamma=1.0, mu=0.
    theta = np.array([[1.0], [0.0]])
    # Initial state covariance
    P = np.eye(2) * 1.0
    # State noise covariance Q and measurement noise variance R_var
    Q = np.eye(2) * delta
    
    # Prepare output series (use the same index as df)
    gamma_series = pd.Series(index=df.index, dtype=np.float64)
    mu_series = pd.Series(index=df.index, dtype=np.float64)
    
    # Kalman filter recursion for each observation
    for t in range(n):
        yt = y.iloc[t]
        xt = x.iloc[t]
        # Measurement matrix: H_t = [xt, 1]
        H = np.array([[xt, 1.0]])
        
        # Prediction step: random walk model so theta_{t|t-1} = theta_{t-1}
        theta_pred = theta
        P = P + Q  # covariance prediction
        
        # Innovation (residual) and innovation covariance
        innovation = yt - (H @ theta_pred)[0, 0]
        S = (H @ P @ H.T)[0, 0] + R_var
        
        # Kalman gain
        K = P @ H.T / S  # shape (2,1)
        
        # Update state estimate
        theta = theta_pred + K * innovation
        
        # Update covariance matrix
        P = (np.eye(2) - K @ H) @ P
        
        # Save current estimates
        gamma_series.iloc[t] = theta[0, 0]
        mu_series.iloc[t] = theta[1, 0]
    
    return gamma_series, mu_series

# -----------------------------
# 4) Pairs Trading & Signal Generation
# -----------------------------
def pairs_trading_strategy(df, gamma, mu, alpha, entry_z=1.5, exit_z=0.0, stop_z=4.0, roll_win=30):
    """
    Compute the trading signal for the pairs trading strategy.
    
    Steps:
      1) Calculate the spread:
         spread_t = log(SpotClose(t)) - gamma(t)*log(FutClose(t)) - mu(t)
      2) Compute a rolling mean/std to obtain the z-score of the spread.
      3) Define entry/exit/stop rules:
           - When zscore > entry_z, consider shorting the spread (unless composite alpha suggests otherwise).
           - When zscore < -entry_z, consider going long.
           - Exit positions when zscore crosses exit_z or if stop-loss conditions are met.
    
    Returns:
      - signal: a pd.Series of positions (+1 for long, -1 for short, 0 for flat)
      - spread: computed spread series
      - zscore: rolling z-score of the spread
    """
    # Compute log prices
    log_spot = np.log(df['SpotClose'] + 1e-9)
    log_fut  = np.log(df['FutClose'] + 1e-9)
    
    # Compute spread at each time point
    spread = log_spot - gamma * log_fut - mu
    spread.name = 'Spread'
    
    # Rolling mean and standard deviation for spread z-score
    spread_mean = spread.rolling(window=roll_win).mean()
    spread_std  = spread.rolling(window=roll_win).std(ddof=1)
    zscore = (spread - spread_mean) / (spread_std + 1e-9)
    
    # Initialize signal series and current position
    signal = pd.Series(data=0.0, index=df.index)
    position = 0  # +1 for long, -1 for short, 0 for flat
    
    # Loop over time (starting after roll_win to ensure valid zscore)
    for i in range(len(zscore)):
        if pd.isna(zscore.iloc[i]):
            signal.iloc[i] = 0
            continue
        
        a_val = alpha.iloc[i]   # composite alpha signal expectation
        z_val = zscore.iloc[i]
        if position == 0:
            # Entry conditions based on spread z-score and alpha adjustment
            if z_val > entry_z:
                # If alpha suggests not to short, skip entry
                if a_val > 0.5:
                    position = 0
                else:
                    position = -1
            elif z_val < -entry_z:
                if a_val < -0.5:
                    position = 0
                else:
                    position = 1
        elif position == 1:
            # Long spread position exit rules
            if z_val > stop_z or z_val > exit_z:
                position = 0
            else:
                position = 1
        elif position == -1:
            # Short spread position exit rules
            if z_val < -stop_z or z_val < exit_z:
                position = 0
            else:
                position = -1
        
        signal.iloc[i] = position
        
    return signal, spread, zscore

# -----------------------------
# 5) PnL Calculation
# -----------------------------
def compute_pnl(df, signal, gamma, mu):
    """
    Calculate performance metrics:
      - Compute the spread, take its daily change, and
      - Compute strategy returns as signal(t-1) * (spread change).
    
    Returns:
      - port_ret: daily returns from the trading strategy
      - cum_pnl: cumulative PnL series from the starting value of 1.0
    """
    log_spot = np.log(df['SpotClose'] + 1e-9)
    log_fut  = np.log(df['FutClose'] + 1e-9)
    spread = log_spot - gamma * log_fut - mu
    spread_ret = spread.diff().fillna(0)
    port_ret = signal.shift(1) * spread_ret
    cum_pnl = (1 + port_ret).cumprod()
    return port_ret, cum_pnl

# -----------------------------
# 6) Main Execution
# -----------------------------
def main():
    # Define file paths (adjust paths as needed)
    futures_csv = "futures_BTCUSDT_1d.csv"
    spots_csv = "spots_BTCUSDT_1d.csv"
    liquidity_csv = "liquid_BTCUSDT_1d.csv"
    
    # 1) Load and merge the data.
    df_all = load_cryptodata(futures_csv, spots_csv, liquidity_csv)
    
    # 2) Compute alpha signals.
    df_alpha = compute_alphas(df_all)
    
    # 3) Estimate dynamic hedge ratios (gamma, mu) via Kalman Filter.
    gamma_series, mu_series = kalman_filter_estimate(df_alpha, delta=1e-4, R_var=1e-3)
    df_alpha['gamma'] = gamma_series
    df_alpha['mu'] = mu_series
    
    # 4) Generate trading signals using pairs trading logic.
    signal, spread, zscore = pairs_trading_strategy(
        df_alpha, gamma_series, mu_series, df_alpha['alpha_composite'],
        entry_z=1.5, exit_z=0.0, stop_z=4.0, roll_win=30
    )
    df_alpha['signal'] = signal
    df_alpha['spread'] = spread
    df_alpha['zscore'] = zscore
    
    # 5) Compute PnL of the trading strategy.
    port_ret, cum_pnl = compute_pnl(df_alpha, signal, gamma_series, mu_series)
    
    # 6) Plot results: Spread, Composite Alpha, and Cumulative PnL.
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    axs[0].plot(df_alpha.index, spread, label='Spread (log(Spot) - γ*log(Fut) - μ)', color='blue')
    axs[0].set_title('Spread (Log-Price Residual)')
    axs[0].legend(loc='best')
    
    axs[1].plot(df_alpha.index, df_alpha['alpha_composite'], label='Composite Alpha', color='magenta')
    axs[1].axhline(0.0, color='gray', linestyle='--')
    axs[1].set_title('Composite Alpha')
    axs[1].legend(loc='best')
    
    axs[2].plot(df_alpha.index, cum_pnl, label='Cumulative PnL', color='green')
    axs[2].set_title('Cumulative PnL (Pairs Trading)')
    axs[2].legend(loc='best')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
