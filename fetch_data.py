"""
Fetch EUR/USD forex data from Yahoo Finance.
Includes retry logic and batching to avoid rate limits.
"""

import os
import time
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from tqdm import tqdm
from utils import load_config, ensure_dirs_exist, retry_with_backoff


def fetch_data_batch(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch data for a specific date range.

    Args:
        symbol: Ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with OHLCV data
    """
    print(f"Fetching {symbol} from {start_date} to {end_date}...")

    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start_date, end=end_date, interval='1h')

    if data.empty:
        raise ValueError(f"No data retrieved for {symbol} in range {start_date} to {end_date}")

    return data


def fetch_eurusd_data(config: dict) -> pd.DataFrame:
    """
    Fetch EUR/USD data from Yahoo Finance with batching and retry logic.

    Args:
        config: Configuration dictionary

    Returns:
        DataFrame with complete EUR/USD data
    """
    symbol = config['data']['symbol']
    batch_days = config['fetching']['batch_days']
    max_retries = config['fetching']['max_retries']
    retry_delay = config['fetching']['retry_delay']

    # Calculate date ranges
    end_date = datetime.now()
    # Fetch maximum available history (Yahoo typically has ~2 years of hourly data)
    start_date = end_date - timedelta(days=730)  # ~2 years

    print(f"\n{'='*60}")
    print(f"Fetching {symbol} data from Yahoo Finance")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print(f"{'='*60}\n")

    all_data = []
    current_date = start_date

    # Create batches
    batches = []
    while current_date < end_date:
        batch_end = min(current_date + timedelta(days=batch_days), end_date)
        batches.append((current_date.strftime('%Y-%m-%d'), batch_end.strftime('%Y-%m-%d')))
        current_date = batch_end

    # Fetch data in batches with progress bar
    for start, end in tqdm(batches, desc="Fetching batches"):
        def fetch_fn():
            return fetch_data_batch(symbol, start, end)

        try:
            # Use retry logic for each batch
            batch_data = retry_with_backoff(
                fetch_fn,
                max_retries=max_retries,
                delay=retry_delay
            )
            all_data.append(batch_data)

            # Small delay between batches to be respectful to Yahoo's servers
            time.sleep(1)

        except Exception as e:
            print(f"\nWarning: Failed to fetch batch {start} to {end}: {str(e)}")
            print("Continuing with available data...")
            continue

    if not all_data:
        raise ValueError("Failed to fetch any data. Please check your internet connection and try again.")

    # Combine all batches
    print("\nCombining batches...")
    df = pd.concat(all_data, axis=0)

    # Remove duplicates and sort by date
    df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()

    # Drop any unnecessary columns
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    print(f"\n{'='*60}")
    print(f"Data fetching completed!")
    print(f"Total rows: {len(df):,}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Columns: {list(df.columns)}")
    print(f"{'='*60}\n")

    return df


def main():
    """Main function to fetch and save EUR/USD data."""
    try:
        # Load configuration
        config = load_config()

        # Ensure directories exist
        ensure_dirs_exist(config)

        # Fetch data
        df = fetch_eurusd_data(config)

        # Save to CSV
        output_path = config['paths']['data_file']
        df.to_csv(output_path)
        print(f"Data saved to: {output_path}")

        # Display sample of the data
        print("\nFirst few rows:")
        print(df.head())
        print("\nLast few rows:")
        print(df.tail())

        # Display basic statistics
        print("\nBasic statistics:")
        print(df.describe())

        # Check for missing values
        missing = df.isnull().sum()
        if missing.any():
            print("\nWarning: Missing values detected:")
            print(missing[missing > 0])
        else:
            print("\nNo missing values detected.")

    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Check your internet connection")
        print("2. Verify that Yahoo Finance is accessible")
        print("3. Try running the script again (sometimes Yahoo has temporary issues)")
        print("4. If the problem persists, check if the symbol 'EURUSD=X' is still valid on Yahoo Finance")
        raise


if __name__ == "__main__":
    main()
