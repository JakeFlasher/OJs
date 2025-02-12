#!/usr/bin/env python3
"""
Combined Strategy Script

This script implements two strategies on two tasks:
 
Task 1: Kimchi Momentum Strategy on daily market data
   - Naive Implementation: Uses fixed threshold rules on Upbit’s daily % changes
     to determine long/short positions on Binance.
   - Advanced Implementation: Uses a rolling LS regression between Binance and
     Upbit returns to estimate a dynamic mispricing spread; a rolling z–score of
     the spread is used to generate trading signals.
     
Task 2: Alpha Factors Trading on hourly data
   - Naive Implementation: Computes three alpha factors (alpha_A, alpha_B, alpha_C)
     and then creates trading signals by averaging the sign of each factor.
   - Advanced Implementation: Computes the same three alphas, then combines them
     via PCA (first principal component extracted) to generate a composite signal.
     
For each task we output graphs to one PDF file:
  - "task1_plots.pdf" for Task 1 (naive then advanced)
  - "task2_plots.pdf" for Task 2 (naive then advanced)
  
Performance metrics (CAGR, max drawdown, Sharpe ratio) are computed in each case.
  
References: Lopez de Prado (2018) and related literature on factor models and backtesting.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm
from sklearn.decomposition import PCA

# --- PERFORMANCE METRICS FUNCTIONS ---
def calculate_cagr(equity_curve, periods_per_year=365):
    initial = equity_curve.iloc[0] if equity_curve.iloc[0] != 0 else 1.0
    total_return = equity_curve.iloc[-1] / initial
    n_periods = len(equity_curve)
    if n_periods == 0:
        return np.nan
    cagr = total_return ** (periods_per_year / n_periods) - 1
    return cagr

def calculate_max_drawdown(equity_curve):
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max
    return drawdown.min()

def calculate_sharpe_ratio(returns, risk_free_rate=0, periods_per_year=365):
    excess_returns = returns - risk_free_rate / periods_per_year
    if returns.std() == 0:
        return np.nan
    return np.sqrt(periods_per_year) * (excess_returns.mean() / returns.std())

# ================================
# TASK 1: MARKET FACTOR STRATEGY
# ================================

# ------ Naive Implementation for Task 1 ------
def naive_load_and_clean_market_data(binance_file, upbit_file):
    # Read Binance data
    df_binance = pd.read_csv(binance_file)
    df_binance['Date'] = pd.to_datetime(df_binance['open_time'], unit='ms')
    df_binance.sort_values('Date', inplace=True)
    df_binance.set_index('Date', inplace=True)
    df_binance = df_binance.rename(columns={'close': 'Binance'})
    # Read Upbit data
    df_upbit = pd.read_csv(upbit_file)
    df_upbit['Date'] = pd.to_datetime(df_upbit['open_time'], unit='ms')
    df_upbit.sort_values('Date', inplace=True)
    df_upbit.set_index('Date', inplace=True)
    df_upbit = df_upbit.rename(columns={'close': 'Upbit'})
    df_merged = pd.concat([df_upbit['Upbit'], df_binance['Binance']], axis=1).dropna()
    return df_merged

def naive_backtest_kimchi_strategy(data, threshold_up, threshold_down):
    returns = []
    for i in range(1, len(data)):
        up_change = (data['Upbit'].iloc[i] - data['Upbit'].iloc[i-1]) / data['Upbit'].iloc[i-1] * 100
        if up_change >= threshold_up:
            position = 1
        elif up_change <= -threshold_down:
            position = -1
        else:
            position = 0
        binance_return = (data['Binance'].iloc[i] - data['Binance'].iloc[i-1]) / data['Binance'].iloc[i-1]
        daily_return = position * binance_return
        returns.append(daily_return)
    ret_series = pd.Series(returns, index=data.index[1:])
    equity_curve = (1 + ret_series).cumprod()
    return ret_series, equity_curve

def naive_task1_process(binance_file, upbit_file, pdf):
    print("Naive Task 1 Processing: Kimchi Momentum Strategy")
    df = naive_load_and_clean_market_data(binance_file, upbit_file)
    # Split data into two halves (backtest and forward test)
    split_index = len(df) // 2
    backtest_data = df.iloc[:split_index]
    print("Backtest period:", backtest_data.index[0].date(), "to", backtest_data.index[-1].date())
    # Grid search thresholds
    threshold_vals = np.arange(0.5, 3.5, 0.5)
    heatmap_df = pd.DataFrame(index=threshold_vals, columns=threshold_vals)
    for X in threshold_vals:
        for Y in threshold_vals:
            strat_ret, _ = naive_backtest_kimchi_strategy(backtest_data, threshold_up=X, threshold_down=Y)
            heatmap_df.loc[X, Y] = calculate_sharpe_ratio(strat_ret)
    heatmap_df = heatmap_df.astype(float)
    
    # Plot heatmap
    plt.figure(figsize=(8,6))
    sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap="viridis")
    plt.title("Naive: Sharpe Ratio Heatmap (Backtest)")
    plt.xlabel("Down Threshold (%)")
    plt.ylabel("Up Threshold (%)")
    pdf.savefig(dpi=600)
    plt.close()
    
    # Use optimal parameters
    optimal_params = heatmap_df.stack().idxmax()
    optimal_X, optimal_Y = optimal_params
    print("Naive Optimal Parameters: Up =", optimal_X, "Down =", optimal_Y)
    
    strat_ret, eq_curve = naive_backtest_kimchi_strategy(backtest_data, threshold_up=optimal_X, threshold_down=optimal_Y)
    cagr = calculate_cagr(eq_curve)
    dd = calculate_max_drawdown(eq_curve)
    sharpe = calculate_sharpe_ratio(strat_ret)
    print(f"Naive Task 1 Metrics: CAGR={cagr:.2%}, Max Drawdown={dd:.2%}, Sharpe={sharpe:.2f}")
    
    # Plot naive equity curve
    plt.figure(figsize=(10,6))
    plt.plot(eq_curve.index, eq_curve, label="Naive Equity Curve")
    plt.title("Naive Task 1: Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    plt.grid(True)
    pdf.savefig(dpi=600)
    plt.close()

# ------ Advanced Implementation for Task 1 ------
# ------ Naive Implementation for Task 1 ------
def load_and_clean_market_data(binance_file, upbit_file):
    # Read Binance data
    df_binance = pd.read_csv(binance_file)
    df_binance['Date'] = pd.to_datetime(df_binance['open_time'], unit='ms')
    df_binance.sort_values('Date', inplace=True)
    df_binance.set_index('Date', inplace=True)
    df_binance = df_binance.rename(columns={'close': 'Binance'})
    # Read Upbit data
    df_upbit = pd.read_csv(upbit_file)
    df_upbit['Date'] = pd.to_datetime(df_upbit['open_time'], unit='ms')
    df_upbit.sort_values('Date', inplace=True)
    df_upbit.set_index('Date', inplace=True)
    df_upbit = df_upbit.rename(columns={'close': 'Upbit'})
    df_merged = pd.concat([df_upbit['Upbit'], df_binance['Binance']], axis=1).dropna()
    return df_merged
def naive_backtest_kimchi_strategy(data, threshold_up, threshold_down):
    """
    Backtest the Kimchi Momentum Strategy in a naive way.
    Trading logic:
      - If Upbit's daily return >= (threshold_up/100), go long Binance.
      - If Upbit's daily return <= -(threshold_down/100), go short Binance.
      - Else, remain flat.
    We use the daily percentage changes as computed by pct_change.
    The signal is shifted by one period to avoid lookahead bias.
    
    Returns:
      - Daily strategy returns (pd.Series)
      - Equity curve (pd.Series)
    """
    # Compute daily returns (decimals)
    up_return = data["Upbit"].pct_change()
    binance_return = data["Binance"].pct_change()
    
    # Generate signal: signal = 1 if up_return >= threshold_up/100, -1 if <= -threshold_down/100, else 0
    signal = pd.Series(0, index=up_return.index)
    signal[up_return >= threshold_up / 100] = 1
    signal[up_return <= -threshold_down / 100] = -1
    
    # Shift signal to avoid lookahead bias
    signal = signal.shift(1)
    
    # Calculate strategy returns based on Binance returns
    strat_returns = signal * binance_return
    strat_returns = strat_returns.fillna(0)
    equity_curve = (1 + strat_returns).cumprod()
    
    return strat_returns, equity_curve

def naive_task1_process(binance_file, upbit_file):
    """
    Process Task 1 using the naive implementation:
      - Load and clean the market data.
      - Split the data into a 50% backtest period.
      - Perform a grid search over thresholds (expressed in percentages)
        to generate a Sharpe ratio heatmap.
      - Compute performance metrics (using 252 trading days per year for daily data).
      - Plot the heatmap and equity curve side by side.
    """
    print("Naive Task 1 Processing: Kimchi Momentum Strategy")
    df = load_and_clean_market_data(binance_file, upbit_file)
    
    # Split data: use first 50% for backtesting
    split_index = len(df) // 2
    backtest_data = df.iloc[:split_index]
    print("Backtest period:", backtest_data.index[0].date(), "to", backtest_data.index[-1].date())
    
    # Grid search over thresholds (values in %)
    threshold_vals = np.arange(0.5, 3.5, 0.5)
    heatmap_df = pd.DataFrame(index=threshold_vals, columns=threshold_vals)
    for X in threshold_vals:
        for Y in threshold_vals:
            strat_ret, _ = naive_backtest_kimchi_strategy(backtest_data, threshold_up=X, threshold_down=Y)
            # Use 252 trading days per year in performance metrics for daily data
            heatmap_df.loc[X, Y] = calculate_sharpe_ratio(strat_ret, periods_per_year=252)
    heatmap_df = heatmap_df.astype(float)
    
    # Create subplots for side-by-side display
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    
    # Left subplot: Sharpe ratio heatmap
    sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap="viridis", ax=axs[0])
    axs[0].set_title("Naive: Sharpe Ratio Heatmap (Backtest)")
    axs[0].set_xlabel(r"Down Threshold (\%)")
    axs[0].set_ylabel(r"Up Threshold (\%)")
    
    # Determine optimal thresholds from the heatmap
    optimal_params = heatmap_df.stack().idxmax()
    optimal_X, optimal_Y = optimal_params
    print("Naive Optimal Parameters: Up =", optimal_X, "Down =", optimal_Y)
    
    # Run backtest using optimal parameters
    strat_ret, eq_curve = naive_backtest_kimchi_strategy(backtest_data, threshold_up=optimal_X, threshold_down=optimal_Y)
    
    cagr = calculate_cagr(eq_curve, periods_per_year=252)
    dd = calculate_max_drawdown(eq_curve)
    sharpe = calculate_sharpe_ratio(strat_ret, periods_per_year=252)
    print(f"Naive Task 1 Metrics: CAGR={cagr:.2%}, Max Drawdown={dd:.2%}, Sharpe={sharpe:.2f}")
    
    # Right subplot: Equity Curve
    axs[1].plot(eq_curve.index, eq_curve, label="Naive Equity Curve", color="magenta")
    axs[1].set_title("Naive Task 1: Equity Curve")
    axs[1].set_xlabel("Date")
    axs[1].set_ylabel("Equity")
    axs[1].legend()
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.show()
    # Save the figure (if needed, using plt.savefig or pdf.savefig with plt.gcf())
    plt.savefig("task1_naive.pdf", dpi=600)

# ------ Advanced Implementation for Task 1 ------
def compute_daily_returns(df):
    return df.pct_change().dropna()

def advanced_rolling_factor_regression(returns, window=30):
    y = returns['Binance']
    X = returns['Upbit']
    X = sm.add_constant(X)
    model = RollingOLS(y, X, window=window)
    rres = model.fit()
    alpha_roll = rres.params['const']
    beta_roll  = rres.params['Upbit']
    return alpha_roll, beta_roll

def advanced_generate_spread_signal(returns, alpha_series, beta_series, roll_window=30, z_threshold=1.75):
    aligned = returns.copy()
    aligned = aligned.join(alpha_series.rename("alpha")).join(beta_series.rename("beta"))
    aligned['model'] = aligned['alpha'] + aligned['beta'] * aligned['Upbit']
    aligned['spread'] = aligned['Binance'] - aligned['model']
    aligned['spread_mean'] = aligned['spread'].rolling(window=roll_window, min_periods=roll_window).mean()
    aligned['spread_std'] = aligned['spread'].rolling(window=roll_window, min_periods=roll_window).std()
    aligned = aligned.dropna()
    aligned['zscore'] = (aligned['spread'] - aligned['spread_mean']) / aligned['spread_std']
    aligned['signal'] = aligned['zscore'].apply(lambda z: -1 if z > z_threshold else (1 if z < -z_threshold else 0))
    return aligned

def advanced_backtest_market_strategy(aligned_data, z_threshold):
    aligned_data['signal_lag'] = aligned_data['signal'].shift(1)
    aligned_data['strategy_returns'] = aligned_data['signal_lag'] * aligned_data['Binance']
    aligned_data['equity_curve'] = (1 + aligned_data['strategy_returns'].fillna(0)).cumprod()
    cagr = calculate_cagr(aligned_data['equity_curve'])
    dd = calculate_max_drawdown(aligned_data['equity_curve'])
    sharpe = calculate_sharpe_ratio(aligned_data['strategy_returns'])
    print("Advanced Task 1 Metrics: CAGR={:.2%}, Max Drawdown={:.2%}, Sharpe={:.2f}".format(cagr, dd, sharpe))
    
    # Plot advanced equity curve and zscore
    fig3 = plt.figure(figsize=(10,5)) 
    plt.subplot(2,1,1)
    plt.plot(aligned_data.index, aligned_data['equity_curve'], label="Advanced Equity Curve", color='blue')
    plt.title("Advanced Task 1: Equity Curve (Rolling Factor Regression)")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2,1,2)
    plt.plot(aligned_data.index, aligned_data['zscore'], label="Spread Z-Score", color='orange')
    plt.axhline(y=z_threshold, color='red', linestyle='--', label="Upper Threshold")
    plt.axhline(y=-z_threshold, color='green', linestyle='--', label="Lower Threshold")
    plt.title("Advanced Task 1: Rolling Z-Score of Mispricing Spread")
    plt.xlabel("Date")
    plt.ylabel("Z-Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig("task1_advanced.pdf", dpi=600)
    plt.close()
    
    return aligned_data

def advanced_task1_process(binance_file, upbit_file):
    print("Advanced Task 1 Processing: Factor Regression–Based Trading")
    df = load_and_clean_market_data(binance_file, upbit_file)
    returns = compute_daily_returns(df)
    window = 30
    alpha_roll, beta_roll = advanced_rolling_factor_regression(returns, window=window)
    z_threshold = 1.75
    aligned = advanced_generate_spread_signal(returns, alpha_roll, beta_roll, roll_window=window, z_threshold=z_threshold)
    advanced_backtest_market_strategy(aligned, z_threshold) 
# ================================
# TASK 2: ALPHA FACTORS STRATEGY
# ================================

# ------ Naive Implementation for Task 2 ------
def naive_load_hourly_data(detail_file):
    df = pd.read_csv(detail_file)
    df['DateTime'] = pd.to_datetime(df['open_time'], unit='ms')
    df.sort_values('DateTime', inplace=True)
    df.set_index('DateTime', inplace=True)
    df.fillna(method='ffill', inplace=True)
    return df

def naive_compute_alpha_factors(df):
    epsilon = 1e-8
    df = df.copy()
    # Compute typical price and VWAP over a rolling 24h window
    df['Typical'] = (df['close'] + df['high'] + df['low']) / 3
    df['VWAP'] = (df['Typical'] * df['volume']).rolling(window=24, min_periods=1).sum() / \
                 df['volume'].rolling(window=24, min_periods=1).sum()
    df['alpha_A'] = np.sqrt(df['high'] * df['low']) - df['VWAP']
    denom = (df['low'] - df['high']) * (df['close']**5) + epsilon
    df['alpha_B'] = - ((df['low'] - df['close']) * (df['open']**5)) / denom
    df['alpha_C'] = (df['close'] - df['open']) / (np.maximum(df['high'] - df['low'], 0.0001) + 0.001)
    return df[['alpha_A', 'alpha_B', 'alpha_C']]

def naive_backtest_alpha_strategy(df):
    df = df.copy()
    df['signal_A'] = np.where(df['alpha_A'] > 0, 1, -1)
    df['signal_B'] = np.where(df['alpha_B'] > 0, 1, -1)
    df['signal_C'] = np.where(df['alpha_C'] > 0, 1, -1)
    combined = (df['signal_A'] + df['signal_B'] + df['signal_C']) / 3
    df['combined_signal'] = np.where(combined >= 0, 1, -1)
    df['returns'] = df['close'].pct_change()
    df['strategy_returns'] = df['combined_signal'].shift(1) * df['returns']
    df['equity_curve'] = (1 + df['strategy_returns'].fillna(0)).cumprod()
    return df

def naive_task2_process(detail_file, pdf):
    print("Naive Task 2 Processing: Alpha Factors Trading")
    df = naive_load_hourly_data(detail_file)
    alphas_df = naive_compute_alpha_factors(df)
    df = naive_backtest_alpha_strategy(pd.concat([df, alphas_df], axis=1))
    cagr = calculate_cagr(df['equity_curve'], periods_per_year=365*24)
    dd = calculate_max_drawdown(df['equity_curve'])
    sharpe = calculate_sharpe_ratio(df['strategy_returns'], periods_per_year=365*24)
    print(f"Naive Task 2 Metrics: CAGR={cagr:.2%}, Max Drawdown={dd:.2%}, Sharpe={sharpe:.2f}")
    
    # Plot naive equity curve
    plt.figure(figsize=(10,6))
    plt.plot(df.index, df['equity_curve'], label="Naive Equity Curve", color='magenta')
    plt.title("Naive Task 2: Equity Curve (Alpha Factors Trading)")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    plt.grid(True)
    pdf.savefig(dpi=600)
    plt.close()

# ------ Advanced Implementation for Task 2 ------
def advanced_load_hourly_data(detail_file):
    # Same cleaning method as naive
    return naive_load_hourly_data(detail_file)

def advanced_compute_alpha_factors(df):
    # Use the same formulas as naive
    return naive_compute_alpha_factors(df)

def advanced_combine_alphas_via_pca(alpha_df, n_components=1):
    alphas = alpha_df.dropna()
    standardized = (alphas - alphas.mean()) / alphas.std()
    pca = PCA(n_components=n_components)
    composite = pca.fit_transform(standardized)
    composite_df = pd.DataFrame(composite, index=standardized.index, columns=['composite_alpha'])
    return composite_df

def advanced_generate_alpha_signal(composite_df):
    df = composite_df.copy()
    # Advanced signal: use sign of composite alpha (a more robust indicator)
    df['signal'] = np.sign(df['composite_alpha'])
    return df

def advanced_backtest_alpha_strategy(price_df, signal_df, composite_df):
    df = price_df.copy()
    df['return'] = df['close'].pct_change()
    signal = signal_df['signal'].reindex(df.index).ffill()
    df['signal'] = signal.shift(1)
    df['strategy_returns'] = df['signal'] * df['return']
    df['equity_curve'] = (1 + df['strategy_returns'].fillna(0)).cumprod()
    cagr = calculate_cagr(df['equity_curve'], periods_per_year=365*24)
    dd = calculate_max_drawdown(df['equity_curve'])
    sharpe = calculate_sharpe_ratio(df['strategy_returns'], periods_per_year=365*24)
    print("Advanced Task 2 Metrics: CAGR={:.2%}, Max Drawdown={:.2%}, Sharpe={:.2f}".format(cagr, dd, sharpe))
    
    plt.figure(figsize=(14,8))
    plt.subplot(2,1,1)
    plt.plot(df.index, df['equity_curve'], label="Advanced Equity Curve", color='blue')
    plt.title("Advanced Task 2: Equity Curve (Composite Alpha Trading)")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2,1,2)
    plt.plot(composite_df.index, composite_df['composite_alpha'], label="Composite Alpha", color='purple')
    plt.axhline(y=0, color='black', linestyle='--', label="Zero")
    plt.title("Advanced Task 2: Composite Alpha Signal")
    plt.xlabel("Date")
    plt.ylabel("Composite Alpha")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    pdf.savefig(dpi=600)
    plt.close()
    
    return df

def advanced_task2_process(detail_file, pdf):
    print("Advanced Task 2 Processing: Composite Alpha Trading")
    df = advanced_load_hourly_data(detail_file)
    alphas_df = advanced_compute_alpha_factors(df)
    composite_df = advanced_combine_alphas_via_pca(alphas_df, n_components=1)
    signal_df = advanced_generate_alpha_signal(composite_df)
    advanced_backtest_alpha_strategy(df, signal_df, composite_df)

# ================================
# MAIN EXECUTION
# ================================
def main():
    # Define file paths (adjust if needed)
    binance_file = 'task_1_data/binance.cleaned.csv'
    upbit_file   = 'task_1_data/upbit_data.cleaned.csv'
    detail_file  = 'task_2_data/binance_1h.csv'
    
    # Produce two PDF files: one for Task 1, one for Task 2.
    with PdfPages("task1_plots.pdf") as pdf1:
        print("\n--- Processing Task 1 (Market Factor Strategy) ---")
        # First, run Naive implementation for Task 1
        naive_task1_process(binance_file, upbit_file, pdf1)
        # Then, run Advanced implementation for Task 1
        advanced_task1_process(binance_file, upbit_file, pdf1)
        print("Task 1 plots saved to 'task1_plots.pdf'")
    
    with PdfPages("task2_plots.pdf") as pdf2:
        print("\n--- Processing Task 2 (Alpha Factors Strategy) ---")
        # First, run Naive implementation for Task 2
        naive_task2_process(detail_file, pdf2)
        # Then, run Advanced implementation for Task 2
        advanced_task2_process(detail_file, pdf2)
        print("Task 2 plots saved to 'task2_plots.pdf'")

if __name__ == "__main__":
    main()