#!/usr/bin/env python3
"""
This script cleans raw CSV data from Binance and Upbit so that both share a unified format.
Input data examples:
--------------------
Binance raw data:
open_time,open,high,low,close,volume,close_time,quote_asset_volume,num_trades,taker_buy_base_asset_volume,taker_buy_quote_asset_volume,ignore,file_date
1609459200000,28948.19,29668.86,28627.12,29337.16,210716.398,1609545599999,6157505024.08511,1511793,101247.902,2960175587.62208,0,2021-01-01
1609545600000,29337.15,33480.0,28958.24,32199.91,545541.08,1609631999999,17122938614.7061,3514545,273388.463,8578964529.70894,0,2021-01-02
Upbit raw data (transformed JSON to CSV):
candleDateTime,candleDateTimeKst,openingPrice,highPrice,lowPrice,tradePrice,candleAccTradeVolume,candleAccTradePrice,timestamp,code,prevClosingPrice,change,changePrice,signedChangePrice,changeRate,signedChangeRate
2025-02-04T00:00:00+00:00,2025-02-04T09:00:00+09:00,101550.23,101551.0,97599.01,99346.57,3.89508841,386569.02867157,1738683639458,CRIX.UPBIT.USDT-BTC,100874.09,FALL,1527.52,-1527.52,0.015142838,-0.015142838
...
The script:
  - Retains the mutual columns: [open_time, open, high, low, close, volume].
  - Computes a new "date" column (in "YYYY-MM-DD" format) from the data timestamp.
  - Writes the cleaned CSV files into the provided output directory with file names ending in ".cleaned.csv".
"""
import argparse
import os
import pandas as pd
def clean_binance(file_path):
    """
    Clean Binance raw CSV data.
    - Retains only the mutual columns:
        open_time, open, high, low, close, volume
    - Converts open_time (already in Unix ms) to a date string ("YYYY-MM-DD")
    """
    # Read Binance data
    df = pd.read_csv(file_path)
    # Define mutual columns
    required_cols = ["open_time", "open", "high", "low", "close", "volume"]
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Binance file {file_path} is missing columns: {missing}")
    df = df[required_cols].copy()
    # Convert open_time to numeric (should already be in ms)
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
    # Ensure other columns are numeric
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Create a new date column with format YYYY-MM-DD from open_time
    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    return df
def clean_upbit(file_path):
    """
    Clean Upbit raw CSV data.
    - Renames columns to match the mutual columns:
         candleDateTime -> open_time
         openingPrice   -> open
         highPrice      -> high
         lowPrice       -> low
         tradePrice     -> close
         candleAccTradeVolume -> volume
    - Retains only these columns.
    - Converts candleDateTime (ISO format) to Unix timestamp in ms,
      and creates a new date column in "YYYY-MM-DD" format.
    """
    # Read Upbit data
    df = pd.read_csv(file_path)
    # Define mapping from Upbit to mutual column names
    rename_map = {
        "candleDateTime": "open_time",
        "openingPrice": "open",
        "highPrice": "high",
        "lowPrice": "low",
        "tradePrice": "close",
        "candleAccTradeVolume": "volume"
    }
    df = df.rename(columns=rename_map)
    # Define required columns
    required_cols = ["open_time", "open", "high", "low", "close", "volume"]
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Upbit file {file_path} is missing columns: {missing}")
    # Retain only required columns
    df = df[required_cols].copy()
    # Convert open_time from ISO8601 to datetime, then to Unix timestamp in ms
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True, errors='coerce')
    df["open_time"] = df["open_time"].view("int64") // 10**6
    # Ensure the numeric fields are properly converted
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Create new date column in "YYYY-MM-DD" format from open_time
    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    return df
def main():
    parser = argparse.ArgumentParser(
        description="Clean Binance and Upbit raw CSV data to a unified format."
    )
    parser.add_argument("binance_path", help="Path to the Binance raw CSV file")
    parser.add_argument("upbit_path", help="Path to the Upbit raw CSV file")
    parser.add_argument("output_dir", help="Directory to write the cleaned CSV files")
    args = parser.parse_args()
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    # Process Binance data
    try:
        binance_df = clean_binance(args.binance_path)
        binance_basename = os.path.basename(args.binance_path)
        # Change extension to ".cleaned.csv"
        binance_output_name = os.path.splitext(binance_basename)[0] + ".cleaned.csv"
        binance_output = os.path.join(args.output_dir, binance_output_name)
        binance_df.to_csv(binance_output, index=False)
        print(f"Cleaned Binance data saved to {binance_output}")
    except Exception as e:
        print(f"Error cleaning Binance file {args.binance_path}: {e}")
    # Process Upbit data
    try:
        upbit_df = clean_upbit(args.upbit_path)
        upbit_basename = os.path.basename(args.upbit_path)
        # Change extension to ".cleaned.csv"
        upbit_output_name = os.path.splitext(upbit_basename)[0] + ".cleaned.csv"
        upbit_output = os.path.join(args.output_dir, upbit_output_name)
        upbit_df.to_csv(upbit_output, index=False)
        print(f"Cleaned Upbit data saved to {upbit_output}")
    except Exception as e:
        print(f"Error cleaning Upbit file {args.upbit_path}: {e}")
if __name__ == '__main__':
    main()
