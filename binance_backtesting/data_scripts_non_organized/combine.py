#!/usr/bin/env python3
"""
This script processes Binance futures daily kline data from local zip files.
Each zip file (named "BTCUSDT-1h-YYYY-MM-DD.zip") is assumed to contain one CSV.
Some CSV files may include header rows; these are automatically detected and skipped.
The script performs the following:
  1. Loops over each zip file in the specified folder.
  2. Reads the CSV data, removes any extra header rows, and limits the data to 12 columns.
  3. Assigns standard column names.
  4. Adds a new "file_date" column derived from the filename.
  5. Concatenates all data into one DataFrame, sorts it by the date and open time, then writes it to a CSV file.
"""

import os
import re
import zipfile
import pandas as pd

# Directory where the zip files are stored
zip_folder = "binance_futures_zips"

# Output CSV file name
output_csv = "combined_BTCUSDT_1h.csv"

# Expected Binance kline columns (12 columns)
# The official Binance kline API returns 12 fields:
# [open_time, open, high, low, close, volume, close_time, quote_asset_volume, num_trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume, ignore]
columns = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "num_trades",
    "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
]

# List to hold DataFrames
dataframes = []

# Regular expression to extract the date from the filename.
# Expected filename: BTCUSDT-1h-YYYY-MM-DD.zip
pattern = re.compile(r"BTCUSDT-1h-(\d{4}-\d{2}-\d{2})\.zip")

# Loop over each zip file in the folder (sorted for reproducibility)
for filename in sorted(os.listdir(zip_folder)):
    if filename.endswith(".zip"):
        match = pattern.search(filename)
        if not match:
            print(f"Filename {filename} does not match expected pattern. Skipping.")
            continue

        file_date_str = match.group(1)
        try:
            file_date = pd.to_datetime(file_date_str)
        except Exception as e:
            print(f"Error converting {file_date_str} to datetime: {e}. Skipping {filename}.")
            continue

        zip_path = os.path.join(zip_folder, filename)
        print(f"Processing {zip_path} ...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                # Find the CSV file(s) in the zip (assuming at least one CSV exists)
                csv_files = [f for f in z.namelist() if f.endswith('.csv')]
                if not csv_files:
                    print(f"No CSV file found in {filename}. Skipping.")
                    continue
                csv_file_name = csv_files[0]

                # Open and read the CSV file; we use header=None so that we can deal with repeated header rows
                with z.open(csv_file_name) as f:
                    df = pd.read_csv(f, header=None)

                # Check if the first row is a header (by comparing first element)
                if isinstance(df.iloc[0, 0], str) and df.iloc[0, 0].lower() == "open_time":
                    df = df.iloc[1:].reset_index(drop=True)

                # Ensure we only take the expected 12 columns (some rows might have an extra header column)
                if df.shape[1] > 12:
                    df = df.iloc[:, :12]

                # Assign the expected column names
                df.columns = columns

                # Add a new column for file_date based on the filename
                df["file_date"] = file_date

                dataframes.append(df)
                print(f"Processed {filename}: {len(df)} rows.")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

if dataframes:
    # Concatenate all DataFrames into one
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Convert open_time column to numeric (if not already) and sort by file_date then open_time.
    # Note: open_time is in milliseconds timestamp.
    combined_df["open_time"] = pd.to_numeric(combined_df["open_time"], errors='coerce')
    combined_df.sort_values(by=["file_date", "open_time"], inplace=True)
    combined_df.reset_index(drop=True, inplace=True)

    # Save the combined data to a CSV file (with a header containing our field names)
    combined_df.to_csv(output_csv, index=False)
    print(f"Combined CSV generated with {len(combined_df)} rows and saved to {output_csv}")
else:
    print("No data was processed. Check your zip files and folder path.")
