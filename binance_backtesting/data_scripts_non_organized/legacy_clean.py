#!/usr/bin/env python3
"""
This script cleans two raw CSV files:
  1. Binance raw data, which has columns such as:
       open_time, open, high, low, close, volume, close_time, quote_asset_volume,
       num_trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume, ignore, file_date
  2. Upbit raw data, which has columns such as:
       candleDateTime, candleDateTimeKst, openingPrice, highPrice, lowPrice, tradePrice,
       candleAccTradeVolume, candleAccTradePrice, timestamp, code, prevClosingPrice,
       change, changePrice, signedChangePrice, changeRate, signedChangeRate

Only the mutual columns are retained and transformed to a common format:
  - open_time   : Unix timestamp in milliseconds.
  - open        : Open price.
  - high        : High price.
  - low         : Low price.
  - close       : For Binance this is the “close” column;
                  for Upbit this is the “tradePrice” column.
  - volume      : For Binance the “volume” column; for Upbit the “candleAccTradeVolume” column.

Usage:
    python clean_data.py path/to/binance.csv path/to/upbit.csv

The output files will be saved with a ".cleaned" suffix (e.g. "binance.csv.cleaned",
 "upbit.csv.cleaned").
"""

import argparse
import os
import pandas as pd

def clean_binance(file_path):
    """
    Clean Binance raw CSV data.
    Retains only the mutual columns and casts open_time to numeric.
    """
    # Read Binance data
    df = pd.read_csv(file_path)
    # Mutual columns from Binance:
    # open_time, open, high, low, close, volume
    required_cols = ["open_time", "open", "high", "low", "close", "volume"]
    
    # Only keep the required columns if they exist
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Binance file {file_path} is missing columns: {missing}")
        
    df = df[required_cols].copy()
    
    # Convert open_time column to numeric (it's already in ms from Binance)
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
    
    # Ensure other columns are numeric
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    return df

def clean_upbit(file_path):
    """
    Clean Upbit raw CSV data.
    Retains the mutual columns and converts candleDateTime (ISO format)
    into a Unix timestamp in milliseconds.
    """
    # Read Upbit data
    df = pd.read_csv(file_path)
    
    # Rename Upbit columns to match our mutual column names:
    # candleDateTime -> open_time
    # openingPrice   -> open
    # highPrice      -> high
    # lowPrice       -> low
    # tradePrice     -> close
    # candleAccTradeVolume -> volume
    rename_map = {
        "candleDateTime": "open_time",
        "openingPrice": "open",
        "highPrice": "high",
        "lowPrice": "low",
        "tradePrice": "close",
        "candleAccTradeVolume": "volume"
    }
    
    df = df.rename(columns=rename_map)
    
    # Check that the needed columns exist
    required_cols = ["open_time", "open", "high", "low", "close", "volume"]
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Upbit file {file_path} is missing columns: {missing}")
    
    # Retain only the mutual columns (even if extra columns are present)
    df = df[required_cols].copy()
    
    # Convert the open_time column from ISO8601 to Unix timestamp (ms)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True, errors='coerce')
    # The dt accessor gives ns; convert to ms.
    df["open_time"] = df["open_time"].view("int64") // 10**6
    
    # Ensure numeric conversion for the other relevant columns
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    return df

def main():
    parser = argparse.ArgumentParser(
        description="Clean Binance and Upbit raw CSV data to a unified format."
    )
    parser.add_argument("binance_path", help="Path to the Binance raw CSV file")
    parser.add_argument("upbit_path", help="Path to the Upbit raw CSV file")
    args = parser.parse_args()
    
    # Process Binance data
    try:
        binance_df = clean_binance(args.binance_path)
        binance_output = args.binance_path + ".cleaned"
        binance_df.to_csv(binance_output, index=False)
        print(f"Cleaned Binance data saved to {binance_output}")
    except Exception as e:
        print(f"Error cleaning Binance file {args.binance_path}: {e}")
    
    # Process Upbit data
    try:
        upbit_df = clean_upbit(args.upbit_path)
        upbit_output = args.upbit_path + ".cleaned"
        upbit_df.to_csv(upbit_output, index=False)
        print(f"Cleaned Upbit data saved to {upbit_output}")
    except Exception as e:
        print(f"Error cleaning Upbit file {args.upbit_path}: {e}")

if __name__ == '__main__':
    main()
