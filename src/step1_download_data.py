"""
=============================================================
Phase 0 | Step 1: Download & Verify TSLA Historical Data
=============================================================
What this script does:
  1. Connects to Alpaca using your API keys from .env
  2. Downloads ~2 months of 5-minute TSLA candles (OHLCV)
  3. Cleans & validates the data
  4. Saves it to data/tsla_5m_historical.csv
  5. Prints a full diagnostic report so you can inspect the data

Run with:
    python src/step1_download_data.py
=============================================================
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# --- Load API keys from .env file ---
load_dotenv()
API_KEY    = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
TICKER     = os.getenv("TICKER", "TSLA")
FEED       = os.getenv("DATA_FEED", "iex")

# ─────────────────────────────────────────────
# VALIDATION: Check that the user filled in .env
# ─────────────────────────────────────────────
def check_keys():
    if not API_KEY or API_KEY == "YOUR_ALPACA_API_KEY_HERE":
        print("\n  ERROR: You haven't set your Alpaca API keys yet!")
        print("  1. Open the file:  .env")
        print("  2. Replace 'YOUR_ALPACA_API_KEY_HERE' with your real key")
        print("  3. Replace 'YOUR_ALPACA_SECRET_KEY_HERE' with your real secret")
        print("  4. Get free keys at: https://app.alpaca.markets (Paper Trading section)")
        sys.exit(1)
    print("  API keys found in .env  ✓")


# ─────────────────────────────────────────────
# STEP 1: Download Historical Bars from Alpaca
# ─────────────────────────────────────────────
def download_historical_data():
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

    # Download last 58 days (Alpaca free tier allows up to 2 years of 1-min data)
    end_date   = datetime.now()
    start_date = end_date - timedelta(days=58)

    print(f"\n  Requesting 5-minute bars for {TICKER}...")
    print(f"  Date range: {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}")
    print(f"  Data feed : {FEED.upper()} (free tier)")

    request = StockBarsRequest(
        symbol_or_symbols=TICKER,
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),   # 5-minute candles
        start=start_date,
        end=end_date,
        feed=FEED
    )

    bars = client.get_stock_bars(request)
    df = bars.df

    # Alpaca returns MultiIndex (symbol, timestamp) — flatten it
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(level=0, drop=True)  # drop the symbol level

    return df


# ─────────────────────────────────────────────
# STEP 2: Clean the Raw Data
# ─────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("\n  Cleaning raw data...")

    # Keep only standard OHLCV columns
    cols_to_keep = ['open', 'high', 'low', 'close', 'volume']
    df = df[[c for c in cols_to_keep if c in df.columns]].copy()

    # Rename to uppercase (standard convention)
    df.columns = [c.capitalize() for c in df.columns]

    before = len(df)
    df.dropna(inplace=True)                    # Remove rows with missing values
    df = df[df['Volume'] > 0]                  # Remove zero-volume candles (holidays/gaps)
    df = df[df['High'] >= df['Low']]           # Sanity check: High must be >= Low
    after = len(df)

    print(f"  Rows before cleaning : {before:,}")
    print(f"  Rows after  cleaning : {after:,}  ({before - after} rows removed)")

    return df


# ─────────────────────────────────────────────
# STEP 3: Validate Data Quality
# ─────────────────────────────────────────────
def validate_data(df: pd.DataFrame):
    print("\n" + "═" * 55)
    print("       DATA QUALITY REPORT")
    print("═" * 55)

    print(f"  Total candles        : {len(df):,}")
    print(f"  Date range           : {df.index.min()} → {df.index.max()}")
    print(f"  Unique trading days  : {df.index.normalize().nunique()}")

    print(f"\n  ── Price Statistics ──")
    print(f"  Close price range    : ${df['Close'].min():.2f} → ${df['Close'].max():.2f}")
    print(f"  Avg candle size(H-L) : ${(df['High'] - df['Low']).mean():.2f} per 5-min candle")
    print(f"  Avg daily volume     : {df['Volume'].mean():,.0f} shares per 5-min bar")

    print(f"\n  ── Missing Data Check ──")
    nulls = df.isnull().sum()
    if nulls.sum() == 0:
        print("  No missing values found ✓ Data is clean!")
    else:
        print(f"  WARNING - Missing values detected:\n{nulls}")

    print(f"\n  ── Sample Candles (first 5 rows) ──")
    print(df.head().to_string())

    print(f"\n  ── Sample Candles (last 5 rows) ──")
    print(df.tail().to_string())

    print("═" * 55)


# ─────────────────────────────────────────────
# STEP 4: Save to CSV
# ─────────────────────────────────────────────
def save_data(df: pd.DataFrame):
    os.makedirs("data", exist_ok=True)
    output_path = "data/tsla_5m_historical.csv"
    df.to_csv(output_path)
    print(f"\n  Data saved to: {output_path}  ✓")
    print(f"  File size    : {os.path.getsize(output_path) / 1024:.1f} KB")
    return output_path


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  TSLA AI INDICATOR — Phase 0: Data Download")
    print("=" * 55)

    check_keys()

    try:
        raw_df = download_historical_data()
        print(f"  Raw data received: {len(raw_df):,} rows ✓")
    except Exception as e:
        print(f"\n  ERROR during download: {e}")
        print("  Common fixes:")
        print("  - Check your API keys are correct in .env")
        print("  - Make sure you are using Paper Trading keys (not Live)")
        sys.exit(1)

    clean_df = clean_data(raw_df)
    validate_data(clean_df)
    path = save_data(clean_df)

    print("\n  Phase 0 Complete! Your dataset is ready.")
    print(f"  Next step: Open '{path}' and explore the data.")
    print("=" * 55)
