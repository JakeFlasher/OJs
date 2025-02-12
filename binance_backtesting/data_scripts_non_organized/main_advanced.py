import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from matplotlib.backends.backend_pdf import PdfPages

#############################################
# Utility Functions for Performance Metrics #
#############################################

def calculate_cagr(equity_curve, periods_per_year=365):
    T = len(equity_curve)
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    return total_return**(periods_per_year / T) - 1

def calculate_max_drawdown(equity_curve):
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max
    return drawdown.min()

def calculate_sharpe_ratio(returns, risk_free_rate=0, periods_per_year=365):
    excess = returns - risk_free_rate/periods_per_year
    return np.sqrt(periods_per_year) * (excess.mean() / excess.std())

###########################################
# Task 1: Advanced Momentum + Forecasting #
###########################################

def process_task1(binance_csv, upbit_csv, pdf):
    # Load and clean data (assumes identical CSV headers: 'open_time', 'close', etc.)
    binance = pd.read_csv(binance_csv)
    upbit = pd.read_csv(upbit_csv)
    
    # Convert Unix timestamps (milliseconds) to datetime
    binance['Date'] = pd.to_datetime(binance['open_time'], unit='ms')
    upbit['Date'] = pd.to_datetime(upbit['open_time'], unit='ms')
    binance.set_index('Date', inplace=True)
    upbit.set_index('Date', inplace=True)
    
    # Use the 'close' price for each exchange and merge on date
    df = pd.concat([upbit['close'].rename('Upbit'), binance['close'].rename('Binance')], axis=1).dropna()
    df = df.asfreq('D')
    # Compute daily returns
    df['Binance_ret'] = df['Binance'].pct_change()
    df['Upbit_ret'] = df['Upbit'].pct_change()
    
    # Split data: first 50% backtest, last 50% forward test
    split = len(df) // 2
    backtest, forward = df.iloc[:split], df.iloc[split:]
    
    # --- Step 1: Momentum Signal from Upbit ---
    # Use thresholds (example: X=1.0%, Y=1.0%) – these can be optimized.
    X, Y = 1.0, 1.0  
    def momentum_signal(ret):
        if ret*100 >= X:
            return 1
        elif ret*100 <= -Y:
            return -1
        else:
            return 0
    backtest['momentum'] = backtest['Upbit_ret'].apply(momentum_signal)
    
    # --- Step 2: AR Forecast on Binance Returns ---
    # Fit an AR(1) model on past Binance_ret (we use the backtest period)
    model = ARIMA(backtest['Binance_ret'].dropna(), order=(1,1,1))
    model_fit = model.fit(method_kwargs={'maxiter': 1000})
    # Forecast next-day return for each day (rolling forecast)
    forecasts = []
    for t in range(20, len(backtest)):  # using a minimum window of 20 obs
        window = backtest['Binance_ret'].iloc[t-20:t]
        try:
            mod = ARIMA(window, order=(1,0,0)).fit()
            fc = mod.forecast(steps=1).iloc[0]
        except Exception:
            fc = np.nan
        forecasts.append(fc)
    # Align forecasts with dates (drop initial days)
    forecast_series = pd.Series(forecasts, index=backtest.index[20:])
    backtest = backtest.iloc[20:]
    backtest['arma_forecast'] = forecast_series
    
    # --- Step 3: GARCH Volatility Forecast on Binance Returns ---
    # Fit a GARCH(1,1) model on the same rolling window (using arch)
    vol_forecasts = []
    for t in range(20, len(backtest)):
        window = backtest['Binance_ret'].iloc[t-20:t]
        try:
            am = arch_model(window, vol='Garch', p=1, q=1, dist='normal')
            res = am.fit(disp='off')
            # Forecast one step ahead
            vol_fc = res.forecast(horizon=1).variance.iloc[-1, 0]**0.5
        except Exception:
            vol_fc = np.nan
        vol_forecasts.append(vol_fc)
    vol_forecast_series = pd.Series(vol_forecasts, index=backtest.index[20:])
    backtest = backtest.iloc[20:]
    backtest['vol_forecast'] = vol_forecast_series
    
    # --- Step 4: Form Composite Signal ---
    # Example: signal = momentum if sign(momentum) == sign(arma_forecast) else 0.
    def composite_signal(row):
        if row['momentum'] == 0 or np.isnan(row['arma_forecast']):
            return 0
        # Only trade if AR prediction agrees with momentum
        if np.sign(row['arma_forecast']) == row['momentum']:
            # Adjust strength scaled inversely by volatility
            weight = 1 if row['vol_forecast']==0 else 1/row['vol_forecast']
            return row['momentum'] * weight
        else:
            return 0
    
    backtest['signal'] = backtest.apply(composite_signal, axis=1)
    
    # --- Step 5: Compute Strategy Returns ---
    # Assume we enter at open/close; here we use Binance_ret as outcome.
    backtest['strategy_ret'] = backtest['signal'].shift(1) * backtest['Binance_ret']
    backtest.dropna(inplace=True)
    
    # Compute performance metrics on backtest
    cagr   = calculate_cagr(backtest['strategy_ret'].add(1).cumprod())
    mdd    = calculate_max_drawdown(backtest['strategy_ret'].add(1).cumprod())
    sharpe = calculate_sharpe_ratio(backtest['strategy_ret'])
    
    print("Task 1 Advanced Strategy Performance (Backtest):")
    print(f"CAGR: {cagr:.2%}, Max Drawdown: {mdd:.2%}, Sharpe Ratio: {sharpe:.2f}")
    
    # Plot equity curve
    plt.figure(figsize=(10,6))
    equity_curve = backtest['strategy_ret'].add(1).cumprod()
    plt.plot(equity_curve.index, equity_curve, label='Strategy Equity')
    plt.title("Advanced Momentum + Forecast Strategy Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    pdf.savefig(dpi=600)
    plt.close()
    
    # (Optimization, further out-of-sample testing, and sensitivity analysis can be added.)

#########################################
# Task 2: Advanced Alpha Factors Strategy #
#########################################

def process_task2(detail_csv, pdf):
    # Load detailed hourly data
    df = pd.read_csv(detail_csv)
    df['DateTime'] = pd.to_datetime(df['open_time'], unit='ms')
    df = df.asfreq('H')
    df.sort_values('DateTime', inplace=True)
    df.set_index('DateTime', inplace=True)
    df.fillna(method='ffill', inplace=True)
    
    # Compute Typical Price and VWAP (rolling 24 hours). Assume hourly data.
    df['Typical'] = (df['close'] + df['high'] + df['low']) / 3
    window = 24
    df['VWAP'] = (df['Typical'] * df['volume']).rolling(window=window, min_periods=1).sum() / \
                 df['volume'].rolling(window=window, min_periods=1).sum()
    
    # Compute alpha factors with edge-case handling:
    epsilon = 1e-8
    df['alpha_A'] = np.sqrt(df['high'] * df['low']) - df['VWAP']
    denominator = (df['low'] - df['high']).replace(0, epsilon) * (df['close']**5 + epsilon)
    df['alpha_B'] = -1 * ((df['low'] - df['close']) * (df['open']**5)) / denominator
    df['alpha_C'] = (df['close'] - df['open']) / (np.maximum(df['high'] - df['low'], 0.0001) + 0.001)
    
    # --- Forecasting Each Alpha with a Simple AR(1) Model ---
    # We use a rolling window forecast for each alpha signal.
    forecast_horizon = 1  # one-step ahead
    for alpha in ['alpha_A', 'alpha_B', 'alpha_C']:
        forecasts = []
        # start forecasting after minimum window (e.g., 24 observations)
        min_window = 24
        for t in range(min_window, len(df)):
            series = df[alpha].iloc[t-min_window:t]
            try:
                model = ARIMA(series, order=(1,0,0)).fit()
                fc = model.forecast(steps=forecast_horizon).iloc[-1]
            except Exception:
                fc = np.nan
            forecasts.append(fc)
        # Align the forecast series
        forecast_series = pd.Series(forecasts, index=df.index[min_window:])
        df.loc[df.index[min_window:], alpha + '_fc'] = forecast_series
    
    # Combine the three forecasted alphas (for example, simple average)
    df['combined_alpha_fc'] = df[['alpha_A_fc', 'alpha_B_fc', 'alpha_C_fc']].mean(axis=1)
    df['alpha_signal'] = np.where(df['combined_alpha_fc'] >= 0, 1, -1)
    
    # Optional: Use a volatility forecast on hourly returns (here we apply a simple rolling std)
    df['hr_ret'] = df['close'].pct_change()
    df['vol_rolling'] = df['hr_ret'].rolling(window=24, min_periods=1).std()
    
    # Define position size: signal scaled inversely to volatility (if volatility is high, reduce position)
    df['position'] = df['alpha_signal'] / df['vol_rolling'].replace(0, np.nan)
    
    # Compute strategy return (using lagged position to avoid lookahead bias)
    df['strategy_ret'] = df['position'].shift(1) * df['hr_ret']
    df.dropna(inplace=True)
    
    # Calculate performance metrics (assume about 24*365 periods/year for hourly data)
    periods_per_year = 24 * 365
    equity_curve = (1 + df['strategy_ret']).cumprod()
    cagr = calculate_cagr(equity_curve, periods_per_year=periods_per_year)
    mdd = calculate_max_drawdown(equity_curve)
    sharpe = calculate_sharpe_ratio(df['strategy_ret'], periods_per_year=periods_per_year)
    
    print("Task 2 Advanced Alpha Factors Strategy Performance:")
    print(f"CAGR: {cagr:.2%}, Max Drawdown: {mdd:.2%}, Sharpe Ratio: {sharpe:.2f}")
    
    # Plot equity curve for Task 2
    plt.figure(figsize=(10,6))
    plt.plot(equity_curve.index, equity_curve, label='Alpha Strategy Equity')
    plt.title("Advanced Alpha Factors Strategy Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    pdf.savefig(dpi=600)
    plt.close()

#######################
# Main Execution Flow #
#######################

def main():
    # File names (adjust as necessary)
    binance_file = 'task_1_data/binance.cleaned.csv'
    upbit_file = 'task_1_data/upbit_data.cleaned.csv'
    detail_file = 'task_2_data/binance_1h.csv'
    
    output_pdf = "advanced_strategies_plots.pdf"
    with PdfPages(output_pdf) as pdf:
        process_task1(binance_file, upbit_file, pdf)
        process_task2(detail_file, pdf)
    
    print(f"\nAll advanced strategy plots have been saved to '{output_pdf}' at 600 DPI.")

if __name__ == "__main__":
    main()
