#!/usr/bin/env python
"""
Robust Advanced Strategy for Quant Developer Test

This script implements a more sophisticated version of two tasks:
   Task 1: A robust momentum-based strategy that combines:
           - A threshold momentum signal from Upbit returns
           - A robust ARIMA forecast of Binance returns (using fallback solvers)
           - A robust GARCH volatility forecast (scaling signal inversely by volatility)
   Task 2: An advanced alpha factor strategy that computes three alpha factors
           (alpha_A, alpha_B, alpha_C), forecasts each using a simple AR(1) model,
           aggregates them into a composite signal, and produces strategy returns.

Performance metrics (CAGR, max drawdown, Sharpe ratio) are computed and equity curves
are plotted for clear visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from matplotlib.backends.backend_pdf import PdfPages
import warnings

#------------------------------------------------------------------
# Performance Metric Functions
#------------------------------------------------------------------
def calculate_cagr(equity_curve, periods_per_year=365):
    T = len(equity_curve)
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    return total_return ** (periods_per_year / T) - 1

def calculate_max_drawdown(equity_curve):
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max
    return drawdown.min()

def calculate_sharpe_ratio(returns, risk_free_rate=0, periods_per_year=365):
    excess = returns - risk_free_rate/periods_per_year
    return np.sqrt(periods_per_year) * (excess.mean() / excess.std())

#------------------------------------------------------------------
# Robust Forecasting Functions
#------------------------------------------------------------------
def robust_arima_forecast(series, order=(1,0,0), forecast_steps=1):
    """
    Attempt to forecast one-step ahead return using ARIMA.
    Tries multiple optimization methods until convergence.
    Returns forecast value (or np.nan if all fail).
    """
    methods = ["lbfgs", "bfgs", "cg"]
    forecast_value = np.nan
    for method in methods:
        try:
            # Exclude missing values in series
            model = ARIMA(series.dropna(), order=order)
            fit = model.fit()
            forecast_value = fit.forecast(steps=forecast_steps).iloc[-1]
            # Check if the model reports convergence via mle_retvals (if available)
            if hasattr(fit, 'mle_retvals'):
                converged = fit.mle_retvals.get("converged", True) or fit.mle_retvals.get("convergence", 0) == 0
                if not converged:
                    raise Exception("Non-convergence with method: " + method)
            break
        except Exception as e:
            warnings.warn(f"ARIMA({order}) failed with method '{method}': {e}")
    return forecast_value

def robust_garch_forecast(series, p=1, q=1, forecast_steps=1):
    """
    Attempt to forecast one-step ahead volatility using a GARCH(p,q) model.
    Returns the forecasted sigma (volatility) for one step.
    """
    try:
        am = arch_model(series.dropna(), vol="Garch", p=p, q=q, dist="normal")
        res = am.fit(disp="off")
        vol_fc = res.forecast(horizon=forecast_steps).variance.iloc[-1, 0] ** 0.5
    except Exception as e:
        warnings.warn(f"GARCH({p},{q}) forecast failed: {e}")
        vol_fc = np.nan
    return vol_fc

#------------------------------------------------------------------
# Task 1: Robust Momentum + Forecast Strategy
#------------------------------------------------------------------
def process_task1(binance_csv, upbit_csv, pdf):
    # Load market CSVs (assumes identical headers including 'open_time','close')
    binance = pd.read_csv(binance_csv)
    upbit = pd.read_csv(upbit_csv)
    
    # Convert Unix timestamps (ms) to datetime and set as index
    binance["Date"] = pd.to_datetime(binance["open_time"], unit="ms")
    upbit["Date"]   = pd.to_datetime(upbit["open_time"], unit="ms")
    binance.set_index("Date", inplace=True)
    upbit.set_index("Date", inplace=True)
    
    # Merge on date using the 'close' price from each exchange
    df = pd.concat([upbit["close"].rename("Upbit"), binance["close"].rename("Binance")], axis=1).dropna()
    df = df.asfreq("D")
    
    # Compute daily returns for each exchange
    df["Binance_ret"] = df["Binance"].pct_change()
    df["Upbit_ret"]   = df["Upbit"].pct_change()
    df.dropna(inplace=True)
    
    # Split data: use first 50% for backtesting
    split = len(df) // 2
    backtest = df.iloc[:split].copy()  # use copy to avoid SettingWithCopyWarning
    
    # ----- Parameter settings -----
    # Set momentum threshold percentages (in percent)
    X_threshold = 1.0  # long if Upbit's return >= 1.0%
    Y_threshold = 1.0  # short if Upbit's return <= -1.0%
    
    # Minimum rolling window for estimation
    window_size = 20

    # Initialize forecast arrays
    arma_forecasts = []
    vol_forecasts  = []
    composite_signals = []  # final signal
    
    # Loop over backtest period (after the window)
    for t in range(window_size, len(backtest)):
        # Get rolling window of Binance returns for ARIMA estimation
        window_data = backtest["Binance_ret"].iloc[t-window_size:t]
        # Robust ARIMA forecast
        forecast_ar = robust_arima_forecast(window_data, order=(1, 0, 0))
        arma_forecasts.append(forecast_ar)
        
        # Robust GARCH forecast of volatility on same window
        vol_fc = robust_garch_forecast(window_data, p=1, q=1)
        vol_forecasts.append(vol_fc)
        
    # Align forecasts with dates (starting at index t = window_size)
    forecast_series = pd.Series(arma_forecasts, index=backtest.index[window_size:])
    vol_series = pd.Series(vol_forecasts, index=backtest.index[window_size:])
    
    # We now compute the momentum signal from Upbit returns based on thresholds:
    def momentum_signal(ret):
        # ret is in decimal; convert to percent
        if ret*100 >= X_threshold:
            return 1
        elif ret*100 <= -Y_threshold:
            return -1
        else:
            return 0
    backtest.loc[:, "momentum"] = backtest["Upbit_ret"].apply(momentum_signal)
    
    # Drop the first window_size observations (as forecasts are not available)
    bt = backtest.iloc[window_size:].copy()
    bt["arma_forecast"] = forecast_series
    bt["vol_forecast"] = vol_series
    
    # Form composite signal: if the momentum signal and sign of AR forecast agree then trade
    # and scale by inverse volatility (if vol_forecast > 0)
    def composite_signal(row):
        if row["momentum"] == 0 or np.isnan(row["arma_forecast"]) or np.isnan(row["vol_forecast"]):
            return 0
        if np.sign(row["arma_forecast"]) == row["momentum"]:
            weight = 1 / row["vol_forecast"] if row["vol_forecast"] > 0 else 1
            return row["momentum"] * weight
        return 0
    bt["signal"] = bt.apply(composite_signal, axis=1)
    
    # Compute strategy returns using next-day position (to avoid lookahead bias)
    bt["strategy_ret"] = bt["signal"].shift(1) * bt["Binance_ret"]
    bt.dropna(inplace=True)
    
    # Compute performance metrics
    equity_curve = (1 + bt["strategy_ret"]).cumprod()
    cagr   = calculate_cagr(equity_curve, periods_per_year=365)
    mdd    = calculate_max_drawdown(equity_curve)
    sharpe = calculate_sharpe_ratio(bt["strategy_ret"], periods_per_year=365)
    
    print("Task 1 Advanced Strategy Performance (Backtest):")
    print(f"CAGR: {cagr:.2%}, Max Drawdown: {mdd:.2%}, Sharpe Ratio: {sharpe:.2f}")
    
    # Plot equity curve
    plt.figure(figsize=(10, 6))
    plt.plot(equity_curve.index, equity_curve, color="darkgreen", lw=2, label="Strategy Equity Curve")
    plt.title("Task 1: Advanced Momentum + Forecast Strategy Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Equity")
    plt.legend()
    plt.grid(True)
    pdf.savefig(dpi=600)
    plt.close()

#------------------------------------------------------------------
# Task 2: Robust Alpha Factors Strategy
#------------------------------------------------------------------
def process_task2(detail_csv, pdf):
    # Load the detailed hourly data
    df = pd.read_csv(detail_csv)
    df["DateTime"] = pd.to_datetime(df["open_time"], unit="ms")
    df.sort_values("DateTime", inplace=True)
    df.set_index("DateTime", inplace=True)
    df = df.asfreq("H")
    df.fillna(method="ffill", inplace=True)
    
    # Compute Typical Price and Rolling VWAP (assume window of 24 hours)
    df["Typical"] = (df["close"] + df["high"] + df["low"]) / 3
    df["VWAP"] = (df["Typical"] * df["volume"]).rolling(window=24, min_periods=1).sum() / \
                 df["volume"].rolling(window=24, min_periods=1).sum()
    
    # Compute alpha factors
    # alpha_A = sqrt(high * low) - VWAP
    df["alpha_A"] = np.sqrt(df["high"] * df["low"]) - df["VWAP"]
    # alpha_B = -((low - close) * (open**5)) / ( (low - high)*(close**5) + 1e-8 )
    epsilon = 1e-8
    df["alpha_B"] = -((df["low"] - df["close"]) * (df["open"]**5)) / ((df["low"] - df["high"])*(df["close"]**5) + epsilon)
    # alpha_C = (close - open) / (max(high - low, 0.0001) + 0.001)
    df["alpha_C"] = (df["close"] - df["open"]) / (np.maximum(df["high"] - df["low"], 0.0001) + 0.001)
    
    # For each alpha factor, forecast one step ahead using a rolling AR(1) model
    forecast_horizon = 1
    min_window = 24  # use at least 24 observations
    
    for alpha_name in ["alpha_A", "alpha_B", "alpha_C"]:
        forecasts = []
        for t in range(min_window, len(df)):
            window_series = df[alpha_name].iloc[t-min_window:t]
            fc = robust_arima_forecast(window_series, order=(1, 0, 0))
            forecasts.append(fc)
        # Align forecast series and add as new column
        forecast_series = pd.Series(forecasts, index=df.index[min_window:])
        df.loc[df.index[min_window:], alpha_name + "_fc"] = forecast_series
    
    # Form composite alpha forecast by averaging the three forecasted alphas
    df["combined_alpha_fc"] = df[["alpha_A_fc", "alpha_B_fc", "alpha_C_fc"]].mean(axis=1)
    
    # Trading signal: use sign of combined forecast (+1 for long, -1 for short)
    df["alpha_signal"] = np.where(df["combined_alpha_fc"] >= 0, 1, -1)
    
    # Compute hourly returns for the close prices
    df["hr_ret"] = df["close"].pct_change()
    # Compute strategy returns using lagged signal (to avoid lookahead bias)
    df["strategy_ret"] = df["alpha_signal"].shift(1) * df["hr_ret"]
    df.dropna(inplace=True)
    
    # Compute performance metrics assuming 24*365 periods per year for hourly data
    periods_per_year = 24 * 365
    equity_curve = (1 + df["strategy_ret"]).cumprod()
    cagr = calculate_cagr(equity_curve, periods_per_year=periods_per_year)
    mdd = calculate_max_drawdown(equity_curve)
    sharpe = calculate_sharpe_ratio(df["strategy_ret"], periods_per_year=periods_per_year)
    
    print("\nTask 2 Advanced Strategy Performance:")
    print(f"CAGR: {cagr:.2%}, Max Drawdown: {mdd:.2%}, Sharpe Ratio: {sharpe:.2f}")
    
    # Plot the equity curve for alpha strategy
    plt.figure(figsize=(10, 6))
    plt.plot(equity_curve.index, equity_curve, color="darkred", lw=2, label="Alpha Strategy Equity Curve")
    plt.title("Task 2: Advanced Alpha Factors Strategy Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Equity")
    plt.legend()
    plt.grid(True)
    pdf.savefig(dpi=600)
    plt.close()

#------------------------------------------------------------------
# Main Execution Flow
#------------------------------------------------------------------
def main():
    # File paths – adjust these paths as needed.
    binance_file = "task_1_data/binance.cleaned.csv"
    upbit_file   = "task_1_data/upbit_data.cleaned.csv"
    detail_file  = "task_2_data/binance_1h.csv"
    
    # Save all plots into one PDF file
    output_pdf = "robust_strategies_plots.pdf"
    with PdfPages(output_pdf) as pdf:
        process_task1(binance_file, upbit_file, pdf)
        process_task2(detail_file, pdf)
    
    print(f"\nAll robust strategy plots have been saved to '{output_pdf}' with 600 DPI.")

if __name__ == "__main__":
    main()
