import numpy as np
import pandas as pd
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

def calculate_cagr(equity_curve, periods_per_year=365):
    """
    Calculate the Compound Annual Growth Rate (CAGR).
    """
    initial = equity_curve.iloc[0] if equity_curve.iloc[0] != 0 else 1.0
    total_return = equity_curve.iloc[-1] / initial
    n_periods = len(equity_curve)
    if n_periods == 0:
        return np.nan
    cagr = total_return ** (periods_per_year / n_periods) - 1
    return cagr

def calculate_max_drawdown(equity_curve):
    """
    Calculate the Maximum Drawdown.
    """
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max
    return drawdown.min()

def calculate_sharpe_ratio(returns, risk_free_rate=0, periods_per_year=252):
    """
    Calculate the annualized Sharpe Ratio.
    Returns are assumed to be in decimal form.
    """
    excess_returns = returns - (risk_free_rate / periods_per_year)
    stdev = excess_returns.std()
    if stdev < 1e-6 or np.isnan(stdev):
        return np.nan
    return np.sqrt(periods_per_year) * (excess_returns.mean() / stdev)

def calculate_calmar_ratio(cagr, max_drawdown):
    """
    Calculate the Calmar Ratio.
    """
    if max_drawdown == 0:
        return np.nan
    return cagr / abs(max_drawdown)
def naive_backtest_kimchi_strategy(data, threshold_up, threshold_down):
    """
    Naive backtesting for Kimchi Momentum Strategy.

    If Upbit's daily return >= (threshold_up/100), go long on Binance.
    If Upbit's daily return <= -(threshold_down/100), go short on Binance.
    Uses shifted signals to avoid lookahead bias.
    """
    # Compute daily percentage returns
    up_return = data["Upbit"].pct_change()
    binance_return = data["Binance"].pct_change()

    # Generate trading signals based on fixed thresholds
    signal = pd.Series(0, index=up_return.index)
    signal[up_return >= threshold_up / 100.0] = 1
    signal[up_return <= -threshold_down / 100.0] = -1

    # Shift the signal to avoid lookahead bias
    signal = signal.shift(1).fillna(0)

    # Calculate strategy returns using Binance returns
    strat_returns = signal * binance_return
    equity_curve = (1 + strat_returns).cumprod()

    return strat_returns, equity_curve

def naive_task1_process(binance_file, upbit_file):
    """
    Process Task 1 using the naive Kimchi Momentum Strategy.
    Loads data, splits the sample (50% backtest), performs a grid search for best parameters,
    computes performance metrics, and plots the heatmap and equity curve.
    """
    print("Naive Task 1 Processing: Kimchi Momentum Strategy")
    df = load_and_clean_market_data(binance_file, upbit_file)

    # Use first 50% as backtest period
    split_index = len(df) // 2
    backtest_data = df.iloc[:split_index]
    print("Backtest period:", backtest_data.index[0].date(), "to", backtest_data.index[-1].date())

    # Grid search for optimal thresholds
    threshold_vals = np.arange(0.5, 3.5, 0.5)
    heatmap_df = pd.DataFrame(index=threshold_vals, columns=threshold_vals)
    for X in threshold_vals:
        for Y in threshold_vals:
            strat_ret, _ = naive_backtest_kimchi_strategy(backtest_data, threshold_up=X, threshold_down=Y)
            heatmap_df.loc[X, Y] = calculate_sharpe_ratio(strat_ret, periods_per_year=252)
    heatmap_df = heatmap_df.astype(float)

    # Plot the Sharpe ratio heatmap and the equity curve
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap="viridis", ax=axs[0])
    axs[0].set_title("Naive: Sharpe Ratio Heatmap (Backtest)")
    axs[0].set_xlabel("Down Threshold (%)")
    axs[0].set_ylabel("Up Threshold (%)")

    # Extract optimal parameters from the grid search
    optimal_params = heatmap_df.stack().idxmax()
    optimal_X, optimal_Y = optimal_params
    print("Naive Optimal Parameters: Up =", optimal_X, "Down =", optimal_Y)

    # Backtest using optimal thresholds
    strat_ret, eq_curve = naive_backtest_kimchi_strategy(backtest_data, threshold_up=optimal_X, threshold_down=optimal_Y)
    cagr = calculate_cagr(eq_curve, periods_per_year=252)
    dd = calculate_max_drawdown(eq_curve)
    sharpe = calculate_sharpe_ratio(strat_ret, periods_per_year=252)
    print(f"Naive Task 1 Metrics: CAGR={cagr:.2%}, Max Drawdown={dd:.2%}, Sharpe={sharpe:.2f}")

    axs[1].plot(eq_curve.index, eq_curve, label="Naive Equity Curve", color="magenta")
    axs[1].set_title("Naive Task 1: Equity Curve")
    axs[1].set_xlabel("Date")
    axs[1].set_ylabel("Equity")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig("naive_sharpe_heatmap.png", dpi=600)
    plt.show()
def compute_daily_returns(df):
    return df.pct_change().dropna()
def advanced_task1_process(binance_file, upbit_file):
    print("Advanced Task 1 Processing: Factor Regression–Based Trading")
    df = load_and_clean_market_data(binance_file, upbit_file)
    returns = compute_daily_returns(df)
    window = 30
    alpha_roll, beta_roll = advanced_rolling_factor_regression(returns, window=window)
    z_threshold = 1.75
    aligned = advanced_generate_spread_signal(returns, alpha_roll, beta_roll, roll_window=window, z_threshold=z_threshold)
    advanced_backtest_market_strategy(aligned, z_threshold) 
def load_hourly_data(detail_file):
    df = pd.read_csv(detail_file)
    df['DateTime'] = pd.to_datetime(df['open_time'], unit='ms')
    df.sort_values('DateTime', inplace=True)
    df.set_index('DateTime', inplace=True)
    # Use .ffill() instead of fillna(method='ffill')
    df = df.ffill()
    return df
def main():
    # Define file names (assume files are provided in the expected CSV format)
    binance_file = 'binance.cleaned.csv'
    upbit_file = 'upbit_data.cleaned.csv'
    detail_file = 'binance_1h.csv'

    # Process Task 1 - Kimchi Momentum Strategies
    naive_task1_process(binance_file, upbit_file)
    advanced_task1_process(binance_file, upbit_file)

    # Process Task 2 - Alpha Factors Strategies
    naive_task2_process(detail_file)
    advanced_task2_process(detail_file)

    # Process Task 3 - Factor-based Arbitrage & Bonus CTA Strategies
    futures_file = "futures_BTCUSDT_1d.csv"
    spots_file = "spots_BTCUSDT_1d.csv"
    liquidity_file = "liquid_BTCUSDT_1d.csv"
    task3_strategy(futures_file, spots_file, liquidity_file, pdf=None)
    cta_strategy_alternative(futures_file, spots_file, pdf=None)

if __name__ == '__main__':
    main()
