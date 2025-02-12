import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Performance Metrics and Plot Functions (from above) ---
def calculate_cagr(equity_curve, periods_per_year=252):
    initial = equity_curve.iloc[0] if equity_curve.iloc[0] != 0 else 1.0
    total_return = equity_curve.iloc[-1] / initial
    n_periods = len(equity_curve)
    if n_periods <= 1:
        return np.nan
    return total_return ** (periods_per_year / n_periods) - 1

def calculate_max_drawdown(equity_curve):
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max
    return drawdown.min()

def calculate_sharpe_ratio(returns, risk_free_rate=0, periods_per_year=252):
    excess_returns = returns - risk_free_rate / periods_per_year
    std = excess_returns.std()
    if std == 0:
        return np.nan
    return (excess_returns.mean() / std) * np.sqrt(periods_per_year)

def calculate_calmar_ratio(cagr, max_drawdown):
    if max_drawdown == 0:
        return np.nan
    return cagr / abs(max_drawdown)

def plot_equity_curve(equity_curve, title, filename):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(equity_curve.index, equity_curve, label="Equity Curve", color="magenta")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(filename, dpi=600)
    plt.show()

# --- Revised Naive Kimchi Momentum Strategy ---
def load_and_clean_market_data(binance_file, upbit_file):
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
    
    df_merged = pd.concat([df_upbit['Upbit'], df_binance['Binance']], axis=1).dropna()
    return df_merged

def naive_backtest_kimchi_strategy(data, threshold_up, threshold_down):
    up_return = data["Upbit"].pct_change()
    binance_return = data["Binance"].pct_change()
    
    # Create signal: 1 when Upbit return >= threshold_up, -1 when <= -threshold_down, 0 otherwise.
    signal = pd.Series(0, index=up_return.index)
    signal[up_return >= threshold_up / 100] = 1
    signal[up_return <= -threshold_down / 100] = -1
    signal = signal.shift(1).fillna(0)
    
    strat_returns = signal * binance_return
    equity_curve = (1 + strat_returns).cumprod()
    return strat_returns, equity_curve

def naive_task1_process(binance_file, upbit_file):
    print("Running Naive Kimchi Momentum Strategy Backtest")
    df = load_and_clean_market_data(binance_file, upbit_file)
    
    # Split data into backtest (first 50%) and forward test if needed.
    split_index = len(df) // 2
    backtest_data = df.iloc[:split_index]
    print("Backtest period: {} to {}".format(backtest_data.index[0].date(), backtest_data.index[-1].date()))
    
    # Grid search for threshold parameters and generate heatmap of Sharpe Ratios.
    threshold_vals = np.arange(0.5, 3.5, 0.5)
    heatmap_df = pd.DataFrame(index=threshold_vals, columns=threshold_vals)
    for up in threshold_vals:
        for down in threshold_vals:
            strat_ret, _ = naive_backtest_kimchi_strategy(backtest_data, threshold_up=up, threshold_down=down)
            heatmap_df.loc[up, down] = calculate_sharpe_ratio(strat_ret, periods_per_year=252)
    
    # Plot Sharpe ratio heatmap.
    plt.figure(figsize=(6, 5))
    sns.heatmap(heatmap_df.astype(float), annot=True, fmt=".2f", cmap="viridis")
    plt.title("Naive Sharpe Heatmap")
    plt.xlabel("Down Threshold (%)")
    plt.ylabel("Up Threshold (%)")
    plt.tight_layout()
    plt.savefig("naive_sharpe_heatmap.png", dpi=600)
    plt.show()
    
    # Determine optimal parameters.
    optimal_params = heatmap_df.stack().idxmax()
    optimal_up, optimal_down = optimal_params
    print("Optimal Up Threshold: {}, Optimal Down Threshold: {}".format(optimal_up, optimal_down))
    
    # Run backtest with optimal parameters.
    strat_ret, eq_curve = naive_backtest_kimchi_strategy(backtest_data, threshold_up=optimal_up, threshold_down=optimal_down)
    cagr = calculate_cagr(eq_curve, periods_per_year=252)
    dd = calculate_max_drawdown(eq_curve)
    sharpe = calculate_sharpe_ratio(strat_ret, periods_per_year=252)
    print("Naive Metrics: CAGR={:.2%}, Max Drawdown={:.2%}, Sharpe={:.2f}".format(cagr, dd, sharpe))
    
    plot_equity_curve(eq_curve, "Naive Kimchi Momentum Equity Curve", "naive_equity_curve.png")

# --- Main Function to Run All Strategies ---
def run_all_strategies():
    # Assumed filenames
    binance_file = 'binance.cleaned.csv'
    upbit_file = 'upbit_data.cleaned.csv'
    
    # Run Task 1: Kimchi Momentum Strategy
    naive_task1_process(binance_file, upbit_file)
    
    # Similar wrappers would be created for advanced Task 1, Task 2 (Alpha Factors),
    # Task 3 (Factor-based arbitrage, bonus CTA, and Kalman filter pairing), etc.
    # Each function should save its plots with predetermined filenames.
    
if __name__ == "__main__":
    run_all_strategies()
