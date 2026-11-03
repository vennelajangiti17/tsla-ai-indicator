"""
=============================================================
Phase 0 | Step 2: Explore & Understand Your TSLA Dataset
=============================================================
What this script does:
  1. Loads your downloaded CSV
  2. Prints a full health report (shape, types, missing values)
  3. Checks for data quality issues (gaps, bad prices)
  4. Plots one full trading day so you can SEE what TSLA looks like
  5. Generates a summary you copy into your NOTES.md

Run with:
    python src/step2_explore_data.py
=============================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os, sys

CSV_PATH = "data/tsla_5m_historical.csv"

# ─────────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────────
def load_data():
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: '{CSV_PATH}' not found.")
        print("Please run step1_download_data.py first.")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)

    # Make sure the index is timezone-aware (convert to New York time)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("America/New_York")

    return df


# ─────────────────────────────────────────────
# Report 1: Basic Info
# ─────────────────────────────────────────────
def report_basic(df):
    print("\n" + "═"*58)
    print("  REPORT 1: Basic Shape & Column Types")
    print("═"*58)
    print(f"  Total rows (candles)  : {len(df):,}")
    print(f"  Total columns         : {len(df.columns)}  → {list(df.columns)}")
    print(f"\n  Column data types:")
    for col, dtype in df.dtypes.items():
        print(f"    {col:<10} → {dtype}")


# ─────────────────────────────────────────────
# Report 2: Date Range & Trading Days
# ─────────────────────────────────────────────
def report_dates(df):
    print("\n" + "═"*58)
    print("  REPORT 2: Date Range & Trading Days")
    print("═"*58)

    first_ts = df.index.min()
    last_ts  = df.index.max()
    trading_days = df.index.normalize().nunique()

    print(f"  First candle : {first_ts}")
    print(f"  Last candle  : {last_ts}")
    print(f"  Trading days : {trading_days} days")

    # Candles per day
    candles_per_day = df.groupby(df.index.date).size()
    print(f"\n  Candles per trading day:")
    print(f"    Expected on a normal day : ~78  (6.5 hours × 60 min ÷ 5 min per candle)")
    print(f"    Min candles in a day     : {candles_per_day.min()} ← (could be a half-day or holiday)")
    print(f"    Max candles in a day     : {candles_per_day.max()}")
    print(f"    Median candles per day   : {candles_per_day.median():.0f}")

    # Show any very short days (holiday/half-day)
    short_days = candles_per_day[candles_per_day < 50]
    if not short_days.empty:
        print(f"\n  ⚠  Short/partial trading days found (< 50 candles):")
        for date, count in short_days.items():
            print(f"    {date} → {count} candles  ← likely a half-day or holiday")
    else:
        print(f"\n  No short trading days found ✓")


# ─────────────────────────────────────────────
# Report 3: Price Statistics
# ─────────────────────────────────────────────
def report_prices(df):
    print("\n" + "═"*58)
    print("  REPORT 3: Price & Volume Statistics")
    print("═"*58)

    candle_size = df['High'] - df['Low']
    print(f"  Close price range   : ${df['Close'].min():.2f}  →  ${df['Close'].max():.2f}")
    print(f"  Average close price : ${df['Close'].mean():.2f}")
    print(f"\n  5-min candle size (High - Low):")
    print(f"    Average : ${candle_size.mean():.2f}")
    print(f"    Max     : ${candle_size.max():.2f}")
    print(f"    Min     : ${candle_size.min():.2f}")
    print(f"\n  Volume (shares per 5-min bar):")
    print(f"    Average : {df['Volume'].mean():,.0f} shares")
    print(f"    Max     : {df['Volume'].max():,.0f} shares  ← a big volume spike like this = news event")
    print(f"    Min     : {df['Volume'].min():,.0f} shares")

    # Flag suspicious candles
    bad_prices = df[(df['Close'] <= 0) | (df['Open'] <= 0)]
    if not bad_prices.empty:
        print(f"\n  ⚠  WARNING: {len(bad_prices)} candles with zero or negative prices found!")
    else:
        print(f"\n  No zero/negative prices found ✓")

    # Detect huge single-candle price jumps (possible data errors)
    pct_change = df['Close'].pct_change().abs()
    spikes = pct_change[pct_change > 0.05]  # > 5% move in one 5-min candle
    if not spikes.empty:
        print(f"\n  ⚠  {len(spikes)} candle(s) with >5% price move (earnings/news events):")
        for ts, val in spikes.head(3).items():
            print(f"    {ts} → {val*100:.1f}% change")
    else:
        print(f"  No extreme price spikes found ✓")


# ─────────────────────────────────────────────
# Report 4: Missing Values
# ─────────────────────────────────────────────
def report_missing(df):
    print("\n" + "═"*58)
    print("  REPORT 4: Missing Values")
    print("═"*58)

    nulls = df.isnull().sum()
    if nulls.sum() == 0:
        print("  No missing (NaN) values found in any column ✓")
    else:
        print("  ⚠  Missing values found:")
        print(nulls[nulls > 0])


# ─────────────────────────────────────────────
# Report 5: Sample Rows
# ─────────────────────────────────────────────
def report_samples(df):
    print("\n" + "═"*58)
    print("  REPORT 5: Sample Data (First 3 & Last 3 Candles)")
    print("═"*58)
    print("\n  -- OLDEST candles in your dataset --")
    print(df.head(3).to_string())
    print("\n  -- MOST RECENT candles in your dataset --")
    print(df.tail(3).to_string())


# ─────────────────────────────────────────────
# Plot: One Full Trading Day
# ─────────────────────────────────────────────
def plot_one_day(df):
    print("\n" + "═"*58)
    print("  CHART: Plotting the most recent trading day...")
    print("═"*58)

    # Get the last full trading day
    last_date = df.index.normalize().unique()[-1]
    day_df = df[df.index.normalize() == last_date].copy()

    if len(day_df) < 5:
        print("  Not enough candles to plot. Skipping chart.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(f'TSLA — 5-Minute Candles — {last_date.date()}', fontsize=14, fontweight='bold')

    # Price chart
    ax1.plot(day_df.index, day_df['Close'], color='#2196F3', linewidth=1.5, label='Close Price')
    ax1.fill_between(day_df.index, day_df['Low'], day_df['High'], alpha=0.15, color='#2196F3', label='High-Low Range')
    ax1.set_ylabel('Price ($)', fontsize=11)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Close Price (each dot = one 5-min candle)', fontsize=10)

    # Volume chart
    ax2.bar(day_df.index, day_df['Volume'], width=0.003, color='#4CAF50', alpha=0.7)
    ax2.set_ylabel('Volume', fontsize=11)
    ax2.set_xlabel('Time (New York)', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax2.set_title('Volume (number of shares traded)', fontsize=10)

    plt.tight_layout()

    os.makedirs("data", exist_ok=True)
    chart_path = "data/tsla_day_chart.png"
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"  Chart saved to: {chart_path}")
    print("  Opening chart...")
    plt.show()


# ─────────────────────────────────────────────
# Notes Summary (copy to NOTES.md)
# ─────────────────────────────────────────────
def print_notes_summary(df):
    trading_days = df.index.normalize().nunique()
    candles_per_day = df.groupby(df.index.date).size()

    print("\n" + "═"*58)
    print("  COPY THIS INTO YOUR NOTES.md")
    print("═"*58)
    print(f"""
## Phase 0 — Data Summary (fill in from the reports above)

- **Ticker**      : TSLA
- **Source**      : Alpaca Markets, IEX feed (Paper Trading account)
- **Timeframe**   : 5-minute candles (OHLCV)
- **Date range**  : {df.index.min().date()} → {df.index.max().date()}
- **Total rows**  : {len(df):,} candles across {trading_days} trading days
- **Avg candles/day** : {candles_per_day.median():.0f}
- **Close range** : ${df['Close'].min():.2f} → ${df['Close'].max():.2f}
- **Missing vals**: None found
- **Timezone**    : America/New_York (9:30 AM – 4:00 PM = IST 8:00 PM – 1:30 AM)
- **Volume note** : IEX feed volume is lower than full SIP market volume.
                    This is fine as long as we train AND run live on IEX feed.
    """)
    print("═"*58)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 58)
    print("  TSLA AI INDICATOR — Phase 0: Explore Data")
    print("=" * 58)

    df = load_data()
    report_basic(df)
    report_dates(df)
    report_prices(df)
    report_missing(df)
    report_samples(df)
    plot_one_day(df)
    print_notes_summary(df)

    print("\n  Phase 0 Step 2 Complete!")
    print("  Next: Run src/step3_live_stream_test.py (during market hours)")
    print("=" * 58)
