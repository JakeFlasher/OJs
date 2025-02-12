"""
Improved Script for Market Data Analysis & Alpha Factors Backtesting

Task 1:
    - Reads Binance and Upbit daily CSV data.
    - Converts Unix timestamps to datetime & aligns data.
    - Splits the data period (50% for backtest, 50% for forward test).
    - Implements the Kimchi Momentum Strategy:
         When Upbit’s price rises X%, go long Binance; when it drops Y%, go short.
    - Loops over a grid of (X, Y) thresholds to generate a Sharpe ratio heatmap.
    - Computes performance metrics (CAGR, Max Drawdown, Sharpe Ratio).

Task 2:
    - Reads detailed hourly trading data from a CSV.
    - Cleans the data and converts Unix timestamps.
    - Computes three alpha factors:
         alpha_A = sqrt(high * low) - VWAP from rolling 24h window,
         alpha_B = -1 * ((low - close) * (open^5)) / (((low - high) * (close^5)) with proper zero handling,
         alpha_C = (close - open) / (max(high - low, 0.0001) + 0.001).
    - Generates simple trading signals based on each alpha and combines them.
    - Runs a backtest of the alpha strategy, then computes the same performance metrics.
    - Plots the performance (equity) curve.

All generated plots are saved into a single PDF file ("analysis_plots.pdf") at 600 DPI.
References include Lopez de Prado (2018) and related literature on backtesting best practices.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
# Set visualization style
# NOTICE: comment this two lines if you do not have scienceplots installed
import scienceplots
plt.style.use(['science','ieee'])
# ------------------ Performance Metrics Functions ------------------

def calculate_cagr(equity_curve, periods_per_year=365):
    """
    Calculate Compound Annual Growth Rate (CAGR).
    """
    n_periods = len(equity_curve)
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    cagr = total_return ** (periods_per_year / n_periods) - 1
    return cagr

def calculate_max_drawdown(equity_curve):
    """
    Calculate Maximum Drawdown.
    """
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max
    return drawdown.min()

def calculate_sharpe_ratio(returns, risk_free_rate=0, periods_per_year=365):
    """
    Calculate the Sharpe Ratio.
    """
    excess_returns = returns - risk_free_rate/periods_per_year
    sharpe = np.sqrt(periods_per_year) * (excess_returns.mean() / excess_returns.std())
    return sharpe

# ------------------ Task 1: Kimchi Momentum Strategy Backtesting ------------------

def load_and_clean_market_data(binance_file, upbit_file):
    """
    Load daily market data from Binance and Upbit CSV files
    and convert Unix timestamps (in ms) to datetime. Returns
    two DataFrames with DateTime index.
    """
    # Read Binance data
    df_binance = pd.read_csv(binance_file)
    df_binance['Date'] = pd.to_datetime(df_binance['open_time'], unit='ms')
    df_binance.sort_values('Date', inplace=True)
    df_binance.set_index('Date', inplace=True)
    
    # Read Upbit data
    df_upbit = pd.read_csv(upbit_file)
    df_upbit['Date'] = pd.to_datetime(df_upbit['open_time'], unit='ms')
    df_upbit.sort_values('Date', inplace=True)
    df_upbit.set_index('Date', inplace=True)
    
    # Rename close price columns for clarity and merge on index (common dates)
    df_binance = df_binance.rename(columns={'close': 'Binance'})
    df_upbit = df_upbit.rename(columns={'close': 'Upbit'})
    df_merged = pd.concat([df_upbit['Upbit'], df_binance['Binance']], axis=1).dropna()
    return df_merged

def backtest_kimchi_strategy(data, threshold_up, threshold_down):
    """
    Backtest the Kimchi Momentum Strategy on daily data.
    Trading logic:
       - If Upbit's daily % change >= threshold_up, go long Binance.
       - If Upbit's daily % change <= -threshold_down, go short Binance.
       - Else, remain flat.
    Returns:
       - Daily strategy returns (pd.Series)
       - Equity curve (pd.Series)
    """
    returns = []
    # Loop through days starting from second day to compute returns
    for i in range(1, len(data)):
        up_change = (data['Upbit'].iloc[i] - data['Upbit'].iloc[i-1]) / data['Upbit'].iloc[i-1] * 100
        if up_change >= threshold_up:
            position = 1
        elif up_change <= -threshold_down:
            position = -1
        else:
            position = 0
        # Binance daily return
        binance_return = (data['Binance'].iloc[i] - data['Binance'].iloc[i-1]) / data['Binance'].iloc[i-1]
        daily_return = position * binance_return
        returns.append(daily_return)
    returns = pd.Series(returns, index=data.index[1:])
    equity_curve = (1 + returns).cumprod()
    return returns, equity_curve

def task1_process(binance_file, upbit_file, pdf):
    """
    Process Task 1:
      - Load and clean market data.
      - Split the data into backtesting period (first 50%) and forward testing (last 50%).
      - Optimize threshold parameters (X & Y) by generating a Sharpe heatmap.
      - Compute performance metrics on the backtest and plot the equity curve.
      - Save the plots to the provided PdfPages object.
    """
    print("Processing Task 1: Market Data Analysis & Kimchi Momentum Strategy")
    df = load_and_clean_market_data(binance_file, upbit_file)
    
    # Split into two halves (backtest and forward)
    split_index = len(df) // 2
    backtest_data = df.iloc[:split_index]
    forward_data = df.iloc[split_index:]
    
    print("Backtest period:", backtest_data.index[0].date(), "to", backtest_data.index[-1].date())
    print("Forward test period:", forward_data.index[0].date(), "to", forward_data.index[-1].date())
    
    # Parameter Optimization: loop over thresholds for X (up) and Y (down)
    threshold_vals = np.arange(0.5, 3.5, 0.5)
    heatmap_df = pd.DataFrame(index=threshold_vals, columns=threshold_vals)
    
    for X in threshold_vals:
        for Y in threshold_vals:
            strat_ret, _ = backtest_kimchi_strategy(backtest_data, threshold_up=X, threshold_down=Y)
            sharpe = calculate_sharpe_ratio(strat_ret)
            heatmap_df.loc[X, Y] = sharpe
            
    heatmap_df = heatmap_df.astype(float)
    
    # Plot the Sharpe ratio heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap="viridis")
    plt.title("Sharpe Ratio Heatmap (Backtest Data)")
    plt.xlabel("Down Threshold (%)")
    plt.ylabel("Up Threshold (%)")
    pdf.savefig(dpi=600)  # save current figure to PDF at 600 DPI
    plt.close()
    
    # Choose optimal parameters (highest Sharpe Ratio)
    optimal_params = heatmap_df.stack().idxmax()
    optimal_X, optimal_Y = optimal_params
    print("Optimal Parameters Found: X (up) =", optimal_X, ", Y (down) =", optimal_Y)
    
    # Backtest using optimal parameters on backtesting data
    opt_ret, opt_eq_curve = backtest_kimchi_strategy(backtest_data, threshold_up=optimal_X, threshold_down=optimal_Y)
    
    # Compute performance metrics
    cagr = calculate_cagr(opt_eq_curve, periods_per_year=365)
    max_dd = calculate_max_drawdown(opt_eq_curve)
    sharpe = calculate_sharpe_ratio(opt_ret, periods_per_year=365)
    
    print("\nTask 1 Performance Metrics (Backtest):")
    print("CAGR: {:.2%}".format(cagr))
    print("Maximum Drawdown: {:.2%}".format(max_dd))
    print("Sharpe Ratio: {:.2f}".format(sharpe))
    
    # Plot Equity Curve for backtest
    plt.figure(figsize=(10, 6))
    plt.plot(opt_eq_curve.index, opt_eq_curve, label="Equity Curve")
    plt.title("Kimchi Momentum Strategy Equity Curve (Backtest)")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    plt.grid(True)
    pdf.savefig(dpi=600)
    plt.close()

# ------------------ Task 2: Alpha Factors Backtesting ------------------

def load_and_clean_detailed_data(detail_file):
    """
    Load detailed trading data from CSV (hourly interval implied by open_time).
    Converts Unix timestamps to datetime and cleans missing values.
    """
    df = pd.read_csv(detail_file)
    df['DateTime'] = pd.to_datetime(df['open_time'], unit='ms')
    df.sort_values('DateTime', inplace=True)
    df.set_index('DateTime', inplace=True)
    # Forward-fill missing values as a precaution.
    df.fillna(method='ffill', inplace=True)
    return df

def compute_alpha_factors(df):
    """
    Compute alpha factors (alpha_A, alpha_B, alpha_C) using provided formulas:
      - alpha_A: sqrt(high * low) - VWAP, with VWAP computed over a rolling 24-hour window.
      - alpha_B: -1 * ((low - close) * (open^5)) / ((low - high) * (close^5)), with small epsilon.
      - alpha_C: (close - open) / (max(high - low, 0.0001) + 0.001)
    """
    # Calculate typical price
    df['Typical'] = (df['close'] + df['high'] + df['low']) / 3
    # Compute rolling VWAP over a 24-hour window (assuming hourly data, window=24)
    df['VWAP'] = (df['Typical'] * df['volume']).rolling(window=24, min_periods=1).sum() \
                 / df['volume'].rolling(window=24, min_periods=1).sum()
    
    # alpha_A calculation:
    df['alpha_A'] = np.sqrt(df['high'] * df['low']) - df['VWAP']
    
    # alpha_B: avoid division by zero by replacing zero denominators with a small value (epsilon)
    epsilon = 1e-8
    denom = (df['low'] - df['high']).replace(0, epsilon) * (df['close']**5 + epsilon)
    df['alpha_B'] = -1 * ((df['low'] - df['close']) * (df['open']**5)) / denom
    
    # alpha_C: use np.maximum to ensure denominator is not too small
    df['alpha_C'] = (df['close'] - df['open']) / (np.maximum(df['high'] - df['low'], 0.0001) + 0.001)
    
    return df

def backtest_alpha_strategy(df):
    """
    Generate simple trading signals based on the computed alpha factors.
    Here, for each alpha we use:
      Signal = +1 if alpha > 0, else -1.
    Combined signal is the average of individual signals (then converted to +1 or -1).
    Backtest strategy: use previous period's signal multiplied by current period's pct change.
    """
    # Create signals
    df['signal_A'] = np.where(df['alpha_A'] > 0, 1, -1)
    df['signal_B'] = np.where(df['alpha_B'] > 0, 1, -1)
    df['signal_C'] = np.where(df['alpha_C'] > 0, 1, -1)
    combined = (df['signal_A'] + df['signal_B'] + df['signal_C']) / 3
    df['combined_signal'] = np.where(combined >= 0, 1, -1)
    
    # Calculate hourly returns from close price
    df['returns'] = df['close'].pct_change()
    # Use previous signal to avoid lookahead bias
    df['strategy_returns'] = df['combined_signal'].shift(1) * df['returns']
    df['equity_curve'] = (1 + df['strategy_returns'].fillna(0)).cumprod()
    return df

def task2_process(detail_file, pdf):
    """
    Process Task 2:
      - Load and clean detailed trading data.
      - Compute alpha factors (A, B, C) and create trading signals.
      - Backtest the combined alpha strategy, compute performance metrics.
      - Plot the equity curve and save the plot to the PDF.
    """
    print("Processing Task 2: Detailed Trading Data & Alpha Factors Backtesting")
    df_detail = load_and_clean_detailed_data(detail_file)
    df_detail = compute_alpha_factors(df_detail)
    df_detail = backtest_alpha_strategy(df_detail)
    
    # Compute performance metrics (for hourly data, assume ~24*365 periods per year)
    periods_per_year = 24 * 365
    strat_returns = df_detail['strategy_returns'].dropna()
    equity_curve = df_detail['equity_curve']
    
    cagr = calculate_cagr(equity_curve, periods_per_year=periods_per_year)
    max_dd = calculate_max_drawdown(equity_curve)
    sharpe = calculate_sharpe_ratio(strat_returns, periods_per_year=periods_per_year)
    
    print("\nTask 2 Performance Metrics:")
    print("CAGR: {:.2%}".format(cagr))
    print("Maximum Drawdown: {:.2%}".format(max_dd))
    print("Sharpe Ratio: {:.2f}".format(sharpe))
    
    # Plot Equity Curve for alpha strategy
    plt.figure(figsize=(10, 6))
    plt.plot(equity_curve.index, equity_curve, label="Alpha Strategy Equity Curve", color='orange')
    plt.title("Alpha Factors Strategy Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    plt.grid(True)
    pdf.savefig(dpi=600)
    plt.close()

# ------------------ Main Execution ------------------

def main():
    # Define file paths (adjust filenames as needed)
    binance_file = 'task_1_data/binance.cleaned.csv'
    upbit_file = 'task_1_data/upbit_data.cleaned.csv'
    detail_file = 'task_2_data/binance_1h.csv'
    
    # Create a PdfPages object to save all plots into a single PDF file.
    output_pdf = "analysis_plots.pdf"
    with PdfPages(output_pdf) as pdf:
        # Process Task 1
        task1_process(binance_file, upbit_file, pdf)
        # Process Task 2
        task2_process(detail_file, pdf)
    
    print("\nAll analysis plots have been saved to '{}' with 600 DPI.".format(output_pdf))

if __name__ == "__main__":
    main()
