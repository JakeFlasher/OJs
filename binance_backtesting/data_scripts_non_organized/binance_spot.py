#!/usr/bin/env python3
"""
This script downloads daily Bitcoin perpetual futures data from Binance's URL,
extracts each CSV from the zip files, and concatenates them into a single DataFrame.

Assumptions:
- The URL is of the form: 
    https://data.binance.vision/data/futures/um/daily/klines/BTCUSDT/1d/BTCUSDT-1d-YYYY-MM-DD.zip
- Each zip file contains one CSV file.
- All CSVs have the same format.

You may customize the date range as needed.
"""

import os
import io
import requests
import zipfile
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy import stats  # For demonstration purposes if any statistical analysis is needed later

# Define the base URL and the output folder for storing temporary zip files (optional)
BASE_URL = "https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1d/"
OUTPUT_DIR = "binance_spot_futures_zips"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define a date range for downloading (this example uses a small range; adjust as needed)
start_date = datetime(2021, 1, 1)
end_date = datetime(2025, 2, 3)
delta = timedelta(days=1)
 
# Loop over each date, download the corresponding zip file, and process it
current_date = start_date
while current_date <= end_date:
    # Format the date as "YYYY-MM-DD"
    date_str = current_date.strftime("%Y-%m-%d")
    zip_filename = f"BTCUSDT-1d-{date_str}.zip"
    download_url = BASE_URL + zip_filename

    print(f"Downloading {download_url} ...")
    response = requests.get(download_url)
    if response.status_code == 200:
        # Optionally, save the zip file locally for debugging or record keeping
        zip_path = os.path.join(OUTPUT_DIR, zip_filename)
        with open(zip_path, "wb") as f_out:
            f_out.write(response.content)
        print(f"Saved zip file to {zip_path}")

        # Open the zip file from memory and extract the CSV data
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # List files inside the zip. We assume there's at least one CSV.
            inner_files = z.namelist()
            print(f"Found files in zip: {inner_files}")
            csv_filename = None
            for f in inner_files:
                if f.endswith(".csv"):
                    csv_filename = f
                    break
            if csv_filename:
                print(f"Extracting and reading {csv_filename} ...")
                with z.open(csv_filename) as csv_file:
                    # Read CSV into a Pandas DataFrame
                    df = pd.read_csv(csv_file)
                    # (Optional) add a column to record the date if needed
                    df["date"] = date_str 
            else:
                print("No CSV file found in the zip.")
    else:
        print(f"Failed to download {download_url}. HTTP status code: {response.status_code}")

    # Move to the next day
    current_date += delta

 