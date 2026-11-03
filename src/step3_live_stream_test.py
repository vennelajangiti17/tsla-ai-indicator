"""
=============================================================
Phase 0 | Step 3: Test the Live Market Data Stream
=============================================================
What this script does:
  - Connects to Alpaca's WebSocket stream
  - Receives LIVE 1-minute TSLA candles as they close
  - Prints each candle in real time with a formatted log
  - Runs for 10 minutes then stops automatically

IMPORTANT — When to run this:
  US Market hours: 9:30 AM – 4:00 PM New York Time
  In IST that is: 8:00 PM – 1:30 AM (IST, UTC+5:30)

  Outside market hours: The script will connect successfully
  but print nothing — that is NORMAL. Run it after 8 PM IST
  on a weekday to see live candles printing.

Run with:
    python src/step3_live_stream_test.py
=============================================================
"""

import os
import asyncio
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
API_KEY    = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
TICKER     = os.getenv("TICKER", "TSLA")

# How many candles to receive before auto-stopping
MAX_CANDLES = 10

# Counter to track received candles
candle_count = 0


def print_banner():
    now_ist = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 60)
    print("  TSLA AI INDICATOR — Live Stream Test")
    print("=" * 60)
    print(f"  Current IST time : {now_ist}")
    print(f"  Ticker           : {TICKER}")
    print(f"  Will auto-stop   : after {MAX_CANDLES} candles received")
    print()
    print("  US Market Hours  : 9:30 AM – 4:00 PM New York Time")
    print("  That is IST      : 8:00 PM – 1:30 AM (next day)")
    print()
    print("  Connecting to Alpaca WebSocket stream...")
    print("  (If no candles appear, market is closed — this is normal)")
    print("-" * 60)


async def on_bar(bar):
    """This function is called automatically each time a new candle closes."""
    global candle_count
    candle_count += 1

    # Format a clean log line for each incoming candle
    ts  = bar.timestamp.strftime("%Y-%m-%d %H:%M") if hasattr(bar.timestamp, 'strftime') else str(bar.timestamp)
    chg = bar.close - bar.open
    direction = "▲ UP  " if chg >= 0 else "▼ DOWN"

    print(f"  [{candle_count:>2}] {ts}  |  "
          f"O:{bar.open:>8.2f}  H:{bar.high:>8.2f}  "
          f"L:{bar.low:>8.2f}  C:{bar.close:>8.2f}  "
          f"Vol:{int(bar.volume):>8,}  |  {direction} ${abs(chg):.2f}")

    if candle_count >= MAX_CANDLES:
        print("-" * 60)
        print(f"  ✓  Received {MAX_CANDLES} candles. Live stream is working!")
        print(f"  ✓  Your data source is confirmed and ready for the live demo.")
        print("=" * 60)
        # Stop the stream after receiving enough candles
        raise KeyboardInterrupt


def main():
    print_banner()

    try:
        from alpaca.data.live import StockDataStream
        from alpaca.data.enums import DataFeed
    except ImportError:
        print("ERROR: alpaca-py not installed. Run: pip install alpaca-py")
        return

    if not API_KEY or API_KEY == "YOUR_ALPACA_API_KEY_HERE":
        print("ERROR: API keys not set. Please fill in your .env file.")
        return

    try:
        stream = StockDataStream(API_KEY, SECRET_KEY, feed=DataFeed.IEX)
        stream.subscribe_bars(on_bar, TICKER)
        print(f"  Connected! Waiting for {TICKER} candles...\n")
        stream.run()

    except KeyboardInterrupt:
        # Clean exit after MAX_CANDLES received (or Ctrl+C from user)
        print("\n  Stream stopped.")

    except Exception as e:
        print(f"\n  ERROR: {e}")
        print("\n  Common fixes:")
        print("  - Check your API keys in .env are correct")
        print("  - Make sure you're using Paper Trading keys (PK...)")
        print("  - You can only run ONE live stream connection at a time")
        print("  - Close any other scripts that might be using the stream")


if __name__ == "__main__":
    main()
