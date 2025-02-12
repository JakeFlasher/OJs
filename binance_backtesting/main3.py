#!/usr/bin/env python
"""
Advanced Robust Strategy for Quant Developer Test

This script implements a more sophisticated version of two tasks by blending
momentum signals, robust ARIMA forecasting, and factor regression ideas (Task 1),
and an advanced alpha factor strategy that processes multiple alphas with 
robust AR(1) forecasts and composite weighting (Task 2).

Key Enhancements:
 - Incorporates a linear factor forecast (regression of Binance returns on Upbit returns)
   in addition to a momentum and ARIMA-based forecast.
 - For Task 2, uses the three alpha factors with robust forecasting and then 
   aggregates them.
 - Optimized plotting and detailed performance evaluation.

Equivalence to R:
 - The factor model formulas from the lecture notes are implemented via np.polyfit.
 - The robust ARIMA and GARCH forecasts remain similar to our previous approach,
   using statsmodels and arch.

Author: OpenAI ChatGPT
Date: 2025-02-04
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
    excess = returns - risk_free_rate / periods_per_year
    return np.sqrt(periods_per_year) * (excess.mean() / excess.std())

#------------------------------------------------------------------
# Robust Forecasting Functions
#------------------------------------------------------------------
def robust_arima_forecast(series, order=(1, 0, 0), forecast_steps=1):
    """
    Forecast one-step ahead using ARIMA.
    Try multiple optimization methods until a forecast is obtained.
    """
    methods = ["lbfgs", "bfgs", "cg"]
    forecast_value = np.nan
    for method in methods:
        try:
            model = ARIMA(series.dropna(), order=order)
            fit = model.fit(method=method)
            forecast_value = fit.forecast(steps=forecast_steps).iloc[-1]
            if hasattr(fit, 'mle_retvals'):
                converged = fit.mle_retvals.get("converged", True) or fit.mle_retvals.get("convergence", 0)==0
                if not converged:
                    raise Exception("Non-convergence with method: " + method)
            break
        except Exception as e:
            warnings.warn(f"ARIMA({order}) failed with method '{method}': {e}")
    return forecast_value

def robust_garch_forecast(series, p=1, q=1, forecast_steps=1):
    """
    Forecast one-step ahead volatility using a GARCH(p,q) model.
    """
    try:
        am = arch_model(series.dropna(), vol="Garch", p=p, q=q, dist="normal")
        res = am.fit(disp="off")
        vol_fc = np.sqrt(res.forecast(horizon=forecast_steps).variance.iloc[-1, 0])
    except Exception as e:
        warnings.warn(f"GARCH({p},{q}) forecast failed: {e}")
        vol_fc = np.nan
    return vol_fc

#------------------------------------------------------------------
# Factor Regression Utility
#------------------------------------------------------------------
def robust_linear_regression(x, y):
    """
    Perform a robust linear regression of y ~ x.
    Returns (intercept, slope) and a forecast:
      forecast = intercept + slope * last_x
    Uses np.polyfit which is OLS; for robust versions one could use RANSAC.
    """
    # Drop NaNs
    mask = (~np.isnan(x)) & (~np.isnan(y))
    if np.sum(mask) < 2:
        return np.nan, np.nan, np.nan
    slope, intercept = np.polyfit(x[mask], y[mask], 1)
    forecast = intercept + slope * x[-1]
    return intercept, slope, forecast

#------------------------------------------------------------------
# Task 1: Advanced Momentum + Forecast Strategy Using Factor Regression
#------------------------------------------------------------------
def process_task1(binance_csv, upbit_csv, pdf):
    # Load CSVs containing daily market data with 'open_time' and 'close'
    binance = pd.read_csv(binance_csv)
    upbit = pd.read_csv(upbit_csv)
    
    # Convert Unix timestamps (ms) to datetime and set as index
    binance["Date"] = pd.to_datetime(binance["open_time"], unit="ms")
    upbit["Date"] = pd.to_datetime(upbit["open_time"], unit="ms")
    binance.set_index("Date", inplace=True)
    upbit.set_index("Date", inplace=True)
    
    # Merge on date using the 'close' price from each exchange and set daily frequency
    df = pd.concat([upbit["close"].rename("Upbit"), binance["close"].rename("Binance")], axis=1).dropna()
    df = df.asfreq("D")
    
    # Compute daily returns for both exchanges
    df["Binance_ret"] = df["Binance"].pct_change()
    df["Upbit_ret"] = df["Upbit"].pct_change()
    df.dropna(inplace=True)
    
    # Split data (first 50% as backtest)
    split = len(df) // 2
    backtest = df.iloc[:split].copy()
    
    # Settings: momentum thresholds in percent, rolling window size, and weights for composite signal.
    X_threshold = 1.0   # Long if Upbit return >= 1.0%
    Y_threshold = 1.0   # Short if Upbit return <= -1.0%
    window_size = 20
    weight_mom = 1.0
    weight_arima = 1.0
    weight_reg   = 1.0

    # Pre-calculate arrays for forecasts (from ARIMA, GARCH, and regression)
    arma_forecasts = []
    vol_forecasts = []
    reg_forecasts = []

    # Loop over the backtest period (after the window)
    for t in range(window_size, len(backtest)):
        # Use rolling window for Binance returns for ARIMA and GARCH
        window_data_binance = backtest["Binance_ret"].iloc[t-window_size:t]
        # ARIMA forecast for Binance return
        arima_fc = robust_arima_forecast(window_data_binance, order=(1, 0, 0))
        arma_forecasts.append(arima_fc)
        # GARCH volatility forecast on the same window
        vol_fc = robust_garch_forecast(window_data_binance, p=1, q=1)
        vol_forecasts.append(vol_fc)
        # Factor regression: regress Binance returns on Upbit returns over window
        window_upbit = backtest["Upbit_ret"].iloc[t-window_size:t].values
        window_binance = window_data_binance.values
        intercept, slope, reg_fc = robust_linear_regression(window_upbit, window_binance)
        reg_forecasts.append(reg_fc)

    # Align forecasts with backtest dates (starting at index = window_size)
    forecast_series = pd.Series(arma_forecasts, index=backtest.index[window_size:])
    vol_series = pd.Series(vol_forecasts, index=backtest.index[window_size:])
    reg_forecast_series = pd.Series(reg_forecasts, index=backtest.index[window_size:])

    # Compute momentum signal from Upbit returns
    def momentum_signal(ret):
        # Convert decimal return to percent comparison
        if ret * 100 >= X_threshold:
            return 1
        elif ret * 100 <= -Y_threshold:
            return -1
        else:
            return 0
    backtest["momentum"] = backtest["Upbit_ret"].apply(momentum_signal)

    # Drop the first `window_size` observations (no forecasts)
    bt = backtest.iloc[window_size:].copy()
    bt["arima_fc"] = forecast_series
    bt["vol_fc"] = vol_series
    bt["reg_fc"] = reg_forecast_series

    # Form composite signal:
    # Normalize ARIMA and regression forecasts by volatility (if available)
    # Then, composite = weighted sum of momentum signal + normalized ARIMA forecast + normalized regression forecast
    def composite_signal(row):
        if (np.isnan(row["arima_fc"]) or np.isnan(row["vol_fc"]) or row["vol_fc"] <= 0 or np.isnan(row["reg_fc"])):
            return 0
        norm_arima = row["arima_fc"] / row["vol_fc"]
        norm_reg = row["reg_fc"] / row["vol_fc"]
        comp = (weight_mom * row["momentum"] +
                weight_arima * np.sign(norm_arima) +
                weight_reg * np.sign(norm_reg))
        return np.sign(comp)
    bt["signal"] = bt.apply(composite_signal, axis=1)

    # Avoid lookahead bias: use previous day's signal for next-day return execution
    bt["strategy_ret"] = bt["signal"].shift(1) * bt["Binance_ret"]
    bt.dropna(inplace=True)

    # Performance evaluation
    equity_curve = (1 + bt["strategy_ret"]).cumprod()
    cagr = calculate_cagr(equity_curve, periods_per_year=365)
    mdd = calculate_max_drawdown(equity_curve)
    sharpe = calculate_sharpe_ratio(bt["strategy_ret"], periods_per_year=365)
    print("Task 1 Advanced Strategy Performance (Backtest):")
    print(f"  CAGR: {cagr:.2%}, Max Drawdown: {mdd:.2%}, Sharpe Ratio: {sharpe:.2f}")

    # Plot equity curve with improved aesthetics
    plt.figure(figsize=(10, 6))
    plt.plot(equity_curve.index, equity_curve, lw=2, color="darkgreen", label="Strategy Equity")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Equity")
    plt.title("Task 1: Advanced Momentum + Forecast Strategy Equity Curve")
    plt.legend()
    plt.grid(alpha=0.4)
    pdf.savefig(dpi=600)
    plt.close()

    # (Optional: Plot individual signals and forecasts for diagnostic purposes)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(bt.index, bt["Binance_ret"], label="Binance Return", color="lightgrey")
    ax.plot(bt.index, bt["arima_fc"], label="ARIMA Forecast", color="blue", alpha=0.7)
    ax.plot(bt.index, bt["reg_fc"], label="Regression Forecast", color="orange", alpha=0.7)
    ax.plot(bt.index, bt["signal"], label="Final Signal", color="red", linestyle="--")
    ax.set_title("Diagnostic: Forecast Components and Final Signal")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value / Signal")
    ax.legend()
    ax.grid(alpha=0.4)
    pdf.savefig(dpi=600)
    plt.close()

#------------------------------------------------------------------
# Task 2: Advanced Alpha Factors Strategy with Composite Forecast
#------------------------------------------------------------------
def process_task2(detail_csv, pdf):
    # Load hourly data and preprocess
    df = pd.read_csv(detail_csv)
    df["DateTime"] = pd.to_datetime(df["open_time"], unit="ms")
    df.sort_values("DateTime", inplace=True)
    df.set_index("DateTime", inplace=True)
    df = df.asfreq("H")
    df.fillna(method="ffill", inplace=True)

    # Compute Typical Price and Rolling VWAP over 24 hours
    df["Typical"] = (df["close"] + df["high"] + df["low"]) / 3
    df["VWAP"] = (df["Typical"] * df["volume"]).rolling(window=24, min_periods=1).sum() / \
                 df["volume"].rolling(window=24, min_periods=1).sum()

    # Compute alpha factors (as given)
    df["alpha_A"] = np.sqrt(df["high"] * df["low"]) - df["VWAP"]
    epsilon = 1e-8
    df["alpha_B"] = -((df["low"] - df["close"]) * (df["open"] ** 5)) / (((df["low"] - df["high"]) * (df["close"] ** 5)) + epsilon)
    df["alpha_C"] = (df["close"] - df["open"]) / (np.maximum(df["high"] - df["low"], 0.0001) + 0.001)

    # Forecast each alpha factor using a rolling AR(1) model with a minimum window (e.g. 24 observations)
    forecast_horizon = 1
    min_window = 24

    for factor in ["alpha_A", "alpha_B", "alpha_C"]:
        forecasts = []
        for t in range(min_window, len(df)):
            history = df[factor].iloc[t-min_window:t]
            fc = robust_arima_forecast(history, order=(1, 0, 0))
            forecasts.append(fc)
        # Align forecast series and add as new column
        forecast_series = pd.Series(forecasts, index=df.index[min_window:])
        df.loc[df.index[min_window:], factor + "_fc"] = forecast_series

    # Composite alpha forecast: for simplicity take the arithmetic mean of the forecasts
    df["composite_alpha_fc"] = df[["alpha_A_fc", "alpha_B_fc", "alpha_C_fc"]].mean(axis=1)

    # Trading signal: +1 if composite forecast nonnegative, -1 otherwise
    df["alpha_signal"] = np.where(df["composite_alpha_fc"] >= 0, 1, -1)

    # Compute hourly returns (log-returns of close price)
    df["hr_ret"] = df["close"].pct_change()
    
    # Use lagged signal to avoid lookahead bias
    df["strategy_ret"] = df["alpha_signal"].shift(1) * df["hr_ret"]
    df.dropna(inplace=True)

    # For hourly data, assume 24*365 periods per year
    periods_per_year = 24 * 365
    equity_curve = (1 + df["strategy_ret"]).cumprod()
    cagr = calculate_cagr(equity_curve, periods_per_year=periods_per_year)
    mdd = calculate_max_drawdown(equity_curve)
    sharpe = calculate_sharpe_ratio(df["strategy_ret"], periods_per_year=periods_per_year)
    print("\nTask 2 Advanced Strategy Performance:")
    print(f"  CAGR: {cagr:.2%}, Max Drawdown: {mdd:.2%}, Sharpe Ratio: {sharpe:.2f}")

    # Plot equity curve for alpha strategy
    plt.figure(figsize=(10, 6))
    plt.plot(equity_curve.index, equity_curve, lw=2, color="darkred", label="Alpha Strategy Equity")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Equity")
    plt.title("Task 2: Advanced Alpha Factors Strategy Equity Curve")
    plt.legend()
    plt.grid(alpha=0.4)
    pdf.savefig(dpi=600)
    plt.close()

    # (Optional: Plot the individual alpha forecasts and composite forecast)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df.index, df["alpha_A_fc"], label="alpha_A_fc", color="blue", alpha=0.7)
    ax.plot(df.index, df["alpha_B_fc"], label="alpha_B_fc", color="green", alpha=0.7)
    ax.plot(df.index, df["alpha_C_fc"], label="alpha_C_fc", color="orange", alpha=0.7)
    ax.plot(df.index, df["composite_alpha_fc"], label="Composite Alpha Forecast", color="purple", linestyle="--")
    ax.set_title("Alpha Forecasts")
    ax.set_xlabel("Date")
    ax.set_ylabel("Forecast Value")
    ax.legend()
    ax.grid(alpha=0.4)
    pdf.savefig(dpi=600)
    plt.close()

#------------------------------------------------------------------
# Main Execution Flow
#------------------------------------------------------------------
def main():
    # File paths (adjust as needed)
    binance_file = "task_1_data/binance.cleaned.csv"
    upbit_file   = "task_1_data/upbit_data.cleaned.csv"
    detail_file  = "task_2_data/binance_1h.csv"
    
    # Save all plots into one PDF file with clear, polished outputs
    output_pdf = "advanced_strategy_plots.pdf"
    with PdfPages(output_pdf) as pdf:
        process_task1(binance_file, upbit_file, pdf)
        process_task2(detail_file, pdf)
    
    print(f"\nAll advanced strategy plots have been saved to '{output_pdf}' with 600 DPI.")

if __name__ == "__main__":
    main()