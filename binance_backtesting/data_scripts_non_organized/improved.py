#!/usr/bin/env python
# improved_trading_strategies.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

# ---------------------------
# Improved Performance Metrics
# ---------------------------
def calculate_cagr(equity_curve, periods_per_year=252):
    """CAGR computed using the first and last equity values."""
    if equity_curve.iloc[0] == 0:
        initial = 1.0
    else:
        initial = equity_curve.iloc[0]
    total_return = equity_curve.iloc[-1] / initial
    n_periods = len(equity_curve)
    if n_periods <= 1:
        return np.nan
    cagr = total_return**(periods_per_year / n_periods) - 1
    return cagr

def calculate_max_drawdown(equity_curve):
    """Maximum drawdown computation."""
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max
    return drawdown.min()

def calculate_sharpe_ratio(returns, risk_free_rate=0, periods_per_year=252):
    """Annualized Sharpe Ratio using percent returns."""
    # Risk free rate is given as annual value; converting to period value.
    excess_returns = returns - (risk_free_rate / periods_per_year)
    std = returns.std()
    if std == 0 or np.isnan(std):
        return np.nan
    sharpe = np.sqrt(periods_per_year) * (excess_returns.mean() / std)
    return sharpe

def calculate_calmar_ratio(cagr, max_drawdown):
    if max_drawdown == 0:
        return np.nan
    return cagr / abs(max_drawdown)

# ---------------------------
# Data Loading Helpers
# ---------------------------
def load_and_clean_market_data(binance_file, upbit_file):
    """Load daily market data and convert prices to returns series."""
    # Read Binance data; assume file contains columns: open_time, close, etc.
    df_binance = pd.read_csv(binance_file)
    df_binance['Date'] = pd.to_datetime(df_binance['open_time'], unit='ms', utc=True)
    df_binance.sort_values('Date', inplace=True)
    df_binance.set_index('Date', inplace=True)
    df_binance = df_binance.rename(columns={'close': 'Binance_Price'})
    
    # Read Upbit data; similar structure.
    df_upbit = pd.read_csv(upbit_file)
    df_upbit['Date'] = pd.to_datetime(df_upbit['open_time'], unit='ms', utc=True)
    df_upbit.sort_values('Date', inplace=True)
    df_upbit.set_index('Date', inplace=True)
    df_upbit = df_upbit.rename(columns={'close': 'Upbit_Price'})
    
    # Merge on date index.
    df_merged = pd.concat([df_upbit['Upbit_Price'], df_binance['Binance_Price']], axis=1).dropna()
    return df_merged

# ---------------------------
# Revised Naive Kimchi Momentum Strategy
# ---------------------------
def naive_backtest_kimchi_strategy(data, threshold_up, threshold_down):
    """
    Backtest using the naive Kimchi Momentum Strategy:
      - Going long Binance if Upbit's pct change >= threshold_up (%)
      - Going short Binance if Upbit's pct change <= -threshold_down (%)
    """
    # Compute percentage returns (decimals)
    up_return = data["Upbit_Price"].pct_change()
    binance_return = data["Binance_Price"].pct_change()
    
    # Generate trading signal based on Upbit change
    signal = pd.Series(0, index=up_return.index)
    signal[up_return >= threshold_up / 100] = 1
    signal[up_return <= -threshold_down / 100] = -1
    
    # Lag the signal to avoid lookahead bias
    signal = signal.shift(1).fillna(0)
    
    # Use percentage returns for strategy
    strat_returns = signal * binance_return.fillna(0)
    equity_curve = (1 + strat_returns).cumprod()
    
    return strat_returns, equity_curve

def naive_task1_process(binance_file, upbit_file):
    print("Processing Naive Kimchi Momentum Strategy...")
    df = load_and_clean_market_data(binance_file, upbit_file)
    
    # Split data 50/50 (backtest vs. forward test)
    split_index = len(df)//2
    backtest_data = df.iloc[:split_index]
    print("Backtest period: {} to {}".format(backtest_data.index[0].date(),
                                               backtest_data.index[-1].date()))
    
    # Grid search for best thresholds
    threshold_vals = np.arange(0.5, 3.5, 0.5)
    heatmap_df = pd.DataFrame(index=threshold_vals, columns=threshold_vals)
    for X in threshold_vals:
        for Y in threshold_vals:
            strat_ret, _ = naive_backtest_kimchi_strategy(backtest_data, threshold_up=X, threshold_down=Y)
            heatmap_df.loc[X, Y] = calculate_sharpe_ratio(strat_ret, periods_per_year=252)
    heatmap_df = heatmap_df.astype(float)
    
    # Plot Sharpe heatmap
    plt.figure(figsize=(8, 4))
    sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap="viridis")
    plt.title("Naive Strategy Sharpe Ratio Heatmap")
    plt.xlabel("Down Threshold (%)")
    plt.ylabel("Up Threshold (%)")
    plt.tight_layout()
    plt.savefig("naive_sharpe_heatmap.png", dpi=300)
    plt.show()
    
    # Choose optimal thresholds based on heatmap (here simply the max cell)
    optimal_params = heatmap_df.stack().idxmax()
    opt_X, opt_Y = optimal_params
    print("Optimal thresholds: Up = {}%, Down = {}%".format(opt_X, opt_Y))
    
    strat_ret, eq_curve = naive_backtest_kimchi_strategy(backtest_data, threshold_up=opt_X, threshold_down=opt_Y)
    cagr = calculate_cagr(eq_curve, periods_per_year=252)
    dd = calculate_max_drawdown(eq_curve)
    sharpe = calculate_sharpe_ratio(strat_ret, periods_per_year=252)
    
    print("Naive Strategy Metrics:")
    print("- CAGR:     {:.2%}".format(cagr))
    print("- Max Drawdown: {:.2%}".format(dd))
    print("- Sharpe Ratio: {:.2f}".format(sharpe))
    
    # Plot equity curve
    plt.figure(figsize=(8, 4))
    plt.plot(eq_curve.index, eq_curve, label="Naive Equity Curve", color="magenta")
    plt.title("Naive Kimchi Momentum Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("naive_equity_curve.png", dpi=300)
    plt.show()

# ---------------------------
# Revised Advanced Kimchi Momentum Strategy
# ---------------------------
def compute_daily_returns(df):
    """Compute daily returns using pct_change for clean return series."""
    return df.pct_change().dropna()

def advanced_rolling_factor_regression(returns, window=30):
    """Perform rolling OLS regression between Binance and Upbit returns."""
    y = returns['Binance_Return']
    X = returns['Upbit_Return']
    X = sm.add_constant(X)
    model = RollingOLS(y, X, window=window)
    rres = model.fit()
    alpha_roll = rres.params['const']
    beta_roll  = rres.params['Upbit_Return']
    return alpha_roll, beta_roll

def advanced_generate_spread_signal(returns, alpha_series, beta_series, roll_window=30, z_threshold=1.75):
    """
    Generate the z–score based signal using a rolling regression spread.
    """
    aligned = returns.copy()
    aligned = aligned.join(alpha_series.rename("alpha")).join(beta_series.rename("beta"))
    # Recalculate model expected return using regression coefficients
    aligned['model'] = aligned['alpha'] + aligned['beta'] * aligned['Upbit_Return']
    aligned['spread'] = aligned['Binance_Return'] - aligned['model']
    aligned['spread_mean'] = aligned['spread'].rolling(window=roll_window, min_periods=roll_window).mean()
    aligned['spread_std'] = aligned['spread'].rolling(window=roll_window, min_periods=roll_window).std()
    aligned = aligned.dropna()
    aligned['zscore'] = (aligned['spread'] - aligned['spread_mean']) / (aligned['spread_std'] + 1e-9)
    aligned['signal'] = aligned['zscore'].apply(lambda z: -1 if z > z_threshold else (1 if z < -z_threshold else 0))
    return aligned

def advanced_backtest_market_strategy(aligned_data, z_threshold):
    """
    Use lagged signals multiplied by daily returns from Binance.
    """
    aligned_data['signal_lag'] = aligned_data['signal'].shift(1).fillna(0)
    # Use percentage return instead of raw price differences.
    aligned_data['strategy_returns'] = aligned_data['signal_lag'] * aligned_data['Binance_Return']
    aligned_data['equity_curve'] = (1 + aligned_data['strategy_returns'].fillna(0)).cumprod()
    cagr = calculate_cagr(aligned_data['equity_curve'], periods_per_year=252)
    dd = calculate_max_drawdown(aligned_data['equity_curve'])
    sharpe = calculate_sharpe_ratio(aligned_data['strategy_returns'], periods_per_year=252)
    print("Advanced Strategy Metrics:")
    print("- CAGR: {:.2%}".format(cagr))
    print("- Max Drawdown: {:.2%}".format(dd))
    print("- Sharpe Ratio: {:.2f}".format(sharpe))
    
    # Plot equity curve and zscore
    fig, axs = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axs[0].plot(aligned_data.index, aligned_data['equity_curve'], label="Advanced Equity Curve", color='blue')
    axs[0].set_title("Advanced Equity Curve (Rolling Regression)")
    axs[0].set_ylabel("Equity")
    axs[0].legend()
    axs[0].grid(True)
    
    axs[1].plot(aligned_data.index, aligned_data['zscore'], label="Spread Z-Score", color='orange')
    axs[1].axhline(y=z_threshold, color='red', linestyle='--', label="Upper Threshold")
    axs[1].axhline(y=-z_threshold, color='green', linestyle='--', label="Lower Threshold")
    axs[1].set_title("Rolling Z-Score")
    axs[1].set_xlabel("Date")
    axs[1].set_ylabel("Z-Score")
    axs[1].legend()
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.savefig("advanced_strategy_plots.png", dpi=300)
    plt.show()
    
    return aligned_data

def advanced_task1_process(binance_file, upbit_file):
    print("Processing Advanced Kimchi Momentum Strategy...")
    df = load_and_clean_market_data(binance_file, upbit_file)
    # Compute returns for both markets
    returns = compute_daily_returns(pd.DataFrame({
        'Binance_Return': df['Binance_Price'].pct_change(),
        'Upbit_Return': df['Upbit_Price'].pct_change()
    }))
    # Use a 30-day rolling window regression.
    window = 30
    alpha_roll, beta_roll = advanced_rolling_factor_regression(returns, window=window)
    z_threshold = 1.75
    aligned = advanced_generate_spread_signal(returns, alpha_roll, beta_roll, roll_window=window, z_threshold=z_threshold)
    advanced_backtest_market_strategy(aligned, z_threshold)

# ---------------------------
# Example Usage and Testing
# ---------------------------
if __name__ == "__main__":
    # Assume we have cleaned CSV files: "binance.cleaned.csv" and "upbit_data.cleaned.csv"
    binance_file = 'binance.cleaned.csv'
    upbit_file = 'upbit_data.cleaned.csv'
    
    # Run Naive Strategy and display plots/metrics.
    naive_task1_process(binance_file, upbit_file)
    
    # Run Advanced Strategy and display plots/metrics.
    advanced_task1_process(binance_file, upbit_file)
    
    # Example expected outputs:
    # - Sharpe heatmap saved as "naive_sharpe_heatmap.png"
    # - Equity curves saved as "naive_equity_curve.png" and "advanced_strategy_plots.png"
    # - Log output showing optimal thresholds and performance metrics