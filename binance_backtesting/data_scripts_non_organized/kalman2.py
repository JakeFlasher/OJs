#!/usr/bin/env python3
"""
Enhanced Task 2 Script using Kalman Filter for Dynamic Hedge Ratio Estimation
with Backtesting Procedure
This script implements an improved pairs trading strategy for BTC/USDT
using daily data. It:
  - Loads merged data from futures, spot, and liquidity CSVs.
  - Computes three alpha factors (momentum, a naive RSI-based, and liquidity rank)
    and forms a composite alpha.
  - Applies a Kalman filter to estimate the dynamic hedge ratio (γ) and intercept (μ)
    in the regression: log(SpotClose) = γ * log(FutClose) + μ + ε.
  - Computes the spread: spread = log(SpotClose) - γ*log(FutClose) - μ.
  - Uses a pairs trading strategy (with entry, exit, and stop-loss rules guided by
    the spread z-score and composite alpha) to generate trading signals.
  - Computes performance metrics and displays three plots:
      1) Spread evolution,
      2) Composite alpha signal, and
      3) Cumulative PnL (full period).
  - Additionally, a backtesting procedure splits the data (e.g., 50-50)
    and measures the equity curves for the backtest period and forward test period.
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
    df_fut = pd.read_csv(futures_csv, parse_dates=['file_date'], index_col='file_date')
    df_spot = pd.read_csv(spots_csv, parse_dates=['file_date'], index_col='file_date')
    df_liq = pd.read_csv(liquidity_csv, parse_dates=['file_date'], index_col='file_date')
    df_liq['Liquidity'] = df_liq['original_quantity'] * df_liq['price']
    # Rename columns for futures to match desired naming: FutOpen, FutHigh, FutLow, FutClose
    df_fut = df_fut.rename(columns={
        'open': 'FutOpen',
        'high': 'FutHigh',
        'low': 'FutLow',
        'close': 'FutClose'
    })
    # Rename columns for spots to match desired naming: SpotOpen, SpotHigh, SpotLow, SpotClose
    df_spot = df_spot.rename(columns={
        'open': 'SpotOpen',
        'high': 'SpotHigh',
        'low': 'SpotLow',
        'close': 'SpotClose'
    })
    # Merge the dataframes by date
    df_all = df_spot.join(df_fut, lsuffix='_spot', rsuffix='_fut', how='inner')
    df_all = df_all.join(df_liq, how='inner')
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
    # Rescale RSI so that larger values indicate oversold (and hence a more attractive long opportunity)
    out['alpha_rsi'] = 1.0 - out['rsi'] / 100.0
    # (C) Liquidity alpha: use dense ranking of the Liquidity column, normalized to [0,1]
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
      - delta: Scalar for state noise covariance (Q = delta * I)
      - R_var: Measurement noise variance
      
    Returns:
      - gamma_series: Estimated dynamic hedge ratios (pd.Series)
      - mu_series: Estimated dynamic intercepts (pd.Series)
    """
    # Convert prices to log-scale
    y = np.log(df['SpotClose'] + 1e-9)
    x = np.log(df['FutClose'] + 1e-9)
    n = len(df)
    # Initialize state vector theta = [gamma, mu] (start with gamma=1.0, mu=0)
    theta = np.array([[1.0], [0.0]])
    # Initial state covariance
    P = np.eye(2) * 1.0
    # State noise covariance Q and measurement noise variance R_var
    Q = np.eye(2) * delta
    gamma_series = pd.Series(index=df.index, dtype=np.float64)
    mu_series = pd.Series(index=df.index, dtype=np.float64)
    # Kalman filter iterations
    for t in range(n):
        yt = y.iloc[t]
        xt = x.iloc[t]
        H = np.array([[xt, 1.0]])  # Measurement matrix
        
        # Prediction step (random walk model)
        theta_pred = theta
        P = P + Q  # Predict state covariance
        
        # Innovation and innovation covariance
        innovation = yt - (H @ theta_pred)[0, 0]
        S = (H @ P @ H.T)[0, 0] + R_var
        
        # Kalman gain
        K = P @ H.T / S  # shape (2,1)
        
        # Update state estimate and covariance
        theta = theta_pred + K * innovation
        P = (np.eye(2) - K @ H) @ P
        
        gamma_series.iloc[t] = theta[0, 0]
        mu_series.iloc[t] = theta[1, 0]
    return gamma_series, mu_series
# -----------------------------
# 4) Pairs Trading & Signal Generation
# -----------------------------
def pairs_trading_strategy(df, gamma, mu, alpha, entry_z=1.5, exit_z=0.0, stop_z=4.0, roll_win=30):
    """
    Compute trading signals based on pairs trading logic.
    1) Calculate the spread: spread_t = log(SpotClose) - gamma(t)*log(FutClose) - mu(t)
    2) Compute a rolling z-score for the spread.
    3) Generate signals:
         - Enter short when z-score > entry_z (if composite alpha does not indicate otherwise).
         - Enter long when z-score < -entry_z.
         - Exit positions when z-score crosses exit_z or when stop-loss conditions occur.
    Returns:
      - signal: pd.Series of positions (+1 for long, -1 for short, 0 for flat)
      - spread: pd.Series of spread values
      - zscore: pd.Series of rolling z-scores
    """
    log_spot = np.log(df['SpotClose'] + 1e-9)
    log_fut  = np.log(df['FutClose'] + 1e-9)
    spread = log_spot - gamma * log_fut - mu
    spread.name = 'Spread'
    spread_mean = spread.rolling(window=roll_win).mean()
    spread_std  = spread.rolling(window=roll_win).std(ddof=1)
    zscore = (spread - spread_mean) / (spread_std + 1e-9)
    signal = pd.Series(data=0.0, index=df.index)
    position = 0  # +1 is long, -1 is short
    for i in range(len(zscore)):
        if pd.isna(zscore.iloc[i]):
            signal.iloc[i] = 0
            continue
        
        a_val = alpha.iloc[i]
        z_val = zscore.iloc[i]
        
        if position == 0:
            if z_val > entry_z:
                if a_val > 0.5:  # alpha suggests against shorting
                    position = 0
                else:
                    position = -1
            elif z_val < -entry_z:
                if a_val < -0.5:  # alpha suggests against going long
                    position = 0
                else:
                    position = 1
        elif position == 1:  # Long position
            if z_val > stop_z or z_val > exit_z:
                position = 0
            else:
                position = 1
        elif position == -1:  # Short position
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
    Compute daily PnL of the strategy and cumulative PnL.
    - Computes the spread and its daily change.
    - Strategy return is the previous period's signal multiplied by the spread change.
    Returns:
      - port_ret: Strategy daily returns.
      - cum_pnl: Cumulative PnL (equity curve).
    """
    log_spot = np.log(df['SpotClose'] + 1e-9)
    log_fut  = np.log(df['FutClose'] + 1e-9)
    spread = log_spot - gamma * log_fut - mu
    spread_ret = spread.diff().fillna(0)
    port_ret = signal.shift(1) * spread_ret
    cum_pnl = (1 + port_ret).cumprod()
    return port_ret, cum_pnl
# -----------------------------
# 6) Backtesting Procedure
# -----------------------------
def backtest_procedure(df_alpha, split_ratio=0.5):
    """
    Split df_alpha (which includes computed signals, gamma, mu, etc.)
    into backtest and forward test periods based on split_ratio.
    Recompute the cumulative equity curves (PnL) for each period.
    Parameters:
      - df_alpha: DataFrame with all strategy signals and computed columns.
      - split_ratio: Proportion of data used for backtesting (default = 0.5).
      
    Returns:
      - cum_pnl_back: Equity curve for the backtest period.
      - cum_pnl_forward: Equity curve for the forward test period.
    """
    T = len(df_alpha)
    split_index = int(T * split_ratio)
    backtest_df = df_alpha.iloc[:split_index].copy()
    forwardtest_df = df_alpha.iloc[split_index:].copy()
    _, cum_pnl_back = compute_pnl(backtest_df, backtest_df['signal'], backtest_df['gamma'], backtest_df['mu'])
    _, cum_pnl_forward = compute_pnl(forwardtest_df, forwardtest_df['signal'], forwardtest_df['gamma'], forwardtest_df['mu'])
    return cum_pnl_back, cum_pnl_forward
# -----------------------------
# 7) Main Execution
# -----------------------------
def main():
    # Define file paths (adjust as needed)
    futures_csv = "futures_BTCUSDT_1d.csv"
    spots_csv = "spots_BTCUSDT_1d.csv"
    liquidity_csv = "liquid_BTCUSDT_1d.csv"
    # 1) Load and merge data
    df_all = load_cryptodata(futures_csv, spots_csv, liquidity_csv)
    # 2) Compute alpha signals
    df_alpha = compute_alphas(df_all)
    # 3) Estimate dynamic hedge ratios (gamma, mu) via Kalman filter
    gamma_series, mu_series = kalman_filter_estimate(df_alpha, delta=1e-4, R_var=1e-3)
    df_alpha['gamma'] = gamma_series
    df_alpha['mu'] = mu_series
    # 4) Generate trading signals based on pairs trading logic
    signal, spread, zscore = pairs_trading_strategy(
        df_alpha, gamma_series, mu_series, df_alpha['alpha_composite'],
        entry_z=1.5, exit_z=0.0, stop_z=4.0, roll_win=30
    )
    df_alpha['signal'] = signal
    df_alpha['spread'] = spread
    df_alpha['zscore'] = zscore
    # 5) Compute overall PnL and plot full-equity curve
    port_ret, cum_pnl = compute_pnl(df_alpha, signal, gamma_series, mu_series)
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    axs[0].plot(df_alpha.index, spread, label='Spread (log(Spot) - γ*log(Fut) - μ)', color='blue')
    axs[0].set_title('Spread (Log-Price Residual)')
    axs[0].legend(loc='best')
    axs[1].plot(df_alpha.index, df_alpha['alpha_composite'], label='Composite Alpha', color='magenta')
    axs[1].axhline(0.0, color='gray', linestyle='--')
    axs[1].set_title('Composite Alpha')
    axs[1].legend(loc='best')
    axs[2].plot(df_alpha.index, cum_pnl, label='Cumulative PnL (Full)', color='green')
    axs[2].set_title('Cumulative PnL (Pairs Trading)')
    axs[2].legend(loc='best')
    plt.tight_layout()
    plt.show()
    # 6) Backtesting procedure: split data into backtest and forward test periods
    cum_pnl_back, cum_pnl_forward = backtest_procedure(df_alpha, split_ratio=0.5)
    fig2, axs2 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axs2[0].plot(cum_pnl_back.index, cum_pnl_back, label='Backtest Equity Curve', color='navy')
    axs2[0].set_title('Equity Curve - Backtest Period')
    axs2[0].set_ylabel('Equity')
    axs2[0].legend(loc='best')
    axs2[1].plot(cum_pnl_forward.index, cum_pnl_forward, label='Forward Test Equity Curve', color='darkorange')
    axs2[1].set_title('Equity Curve - Forward Test Period')
    axs2[1].set_xlabel('Date')
    axs2[1].set_ylabel('Equity')
    axs2[1].legend(loc='best')
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    main()
