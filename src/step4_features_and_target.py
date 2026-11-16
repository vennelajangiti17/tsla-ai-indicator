"""
=============================================================
Phase 1 | Step 4: Feature Engineering & Target Definition
=============================================================
What this script does:
  1. Defines the prediction TARGET (what the model predicts)
  2. Computes 20+ technical indicator FEATURES from raw OHLCV
  3. Performs a safe CHRONOLOGICAL train/val/test split
  4. Saves the final ML-ready dataset to data/tsla_features.csv
  5. Measures BASELINE accuracy (what a dumb model scores)
     → Your real model MUST beat this number

Run with:
    python src/step4_features_and_target.py
=============================================================
"""

import pandas as pd
import numpy as np
import ta          # Technical Analysis library
import os, sys

CSV_PATH  = "data/tsla_5m_historical.csv"
OUT_PATH  = "data/tsla_features.csv"

# ─────────────────────────────────────────────────────────
# TARGET SETTINGS  (the core design decision)
# ─────────────────────────────────────────────────────────
# We predict: "Will the close price be HIGHER than now
#              after LOOKAHEAD_CANDLES candles?"
#
# LOOKAHEAD_CANDLES = 3 → predicting 15 minutes ahead
#   (3 candles × 5 min = 15 min)
#
# MIN_MOVE_PCT = 0.10%  → if price moves less than 0.10%,
#   we label it "flat" and DROP that row (avoids training
#   on random noise where direction doesn't matter)
#
# This gives us a BINARY target:
#   1 = price goes UP by >0.10% in next 15 min
#   0 = price goes DOWN by >0.10% in next 15 min
# ─────────────────────────────────────────────────────────
LOOKAHEAD_CANDLES = 2    # predict 10 minutes ahead (2 × 5min)
MIN_MOVE_PCT      = 0.08 # slightly lower threshold to keep more rows

# Train/Val/Test split ratios (MUST be chronological, NEVER random)
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO  = 0.15 (remainder)


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
def load_data():
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: '{CSV_PATH}' not found. Run step1 first.")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("America/New_York")
    df = df.sort_index()  # ensure chronological order
    print(f"  Loaded {len(df):,} candles from {CSV_PATH}")
    return df


# ─────────────────────────────────────────────
# STEP 1: DEFINE THE TARGET
# ─────────────────────────────────────────────
def add_target(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n  Defining target: price direction in next "
          f"{LOOKAHEAD_CANDLES} candles ({LOOKAHEAD_CANDLES*5} min)...")

    # Future close price (N candles ahead)
    df['future_close'] = df['Close'].shift(-LOOKAHEAD_CANDLES)

    # Percentage change from current close to future close
    df['future_pct_change'] = (df['future_close'] - df['Close']) / df['Close'] * 100

    # Binary target:
    #   1 = price will go UP   (future_pct_change > +MIN_MOVE_PCT)
    #   0 = price will go DOWN (future_pct_change < -MIN_MOVE_PCT)
    #   NaN = "flat" row — will be dropped
    df['target'] = np.nan
    df.loc[df['future_pct_change'] >  MIN_MOVE_PCT, 'target'] = 1   # UP
    df.loc[df['future_pct_change'] < -MIN_MOVE_PCT, 'target'] = 0   # DOWN
    # "flat" rows (between -0.10% and +0.10%) stay NaN → dropped later

    up_count   = (df['target'] == 1).sum()
    down_count = (df['target'] == 0).sum()
    flat_count = df['target'].isna().sum()

    print(f"\n  Target distribution (before dropping flat rows):")
    print(f"    UP   (1) : {up_count:,}  ({up_count/len(df)*100:.1f}%)")
    print(f"    DOWN (0) : {down_count:,}  ({down_count/len(df)*100:.1f}%)")
    print(f"    FLAT/NaN : {flat_count:,}  ({flat_count/len(df)*100:.1f}%)  ← these will be dropped")

    return df


# ─────────────────────────────────────────────
# STEP 2: COMPUTE FEATURES (Technical Indicators)
# ─────────────────────────────────────────────
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n  Computing technical indicator features...")

    close  = df['Close']
    high   = df['High']
    low    = df['Low']
    volume = df['Volume']

    # ── TREND indicators ──────────────────────────
    # RSI: measures if a stock is overbought (>70) or oversold (<30)
    df['rsi_14'] = ta.momentum.RSIIndicator(close, window=14).rsi()

    # MACD: measures momentum by comparing two moving averages
    macd_obj = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df['macd']        = macd_obj.macd()           # MACD line
    df['macd_signal'] = macd_obj.macd_signal()    # Signal line
    df['macd_hist']   = macd_obj.macd_diff()      # Histogram (MACD - Signal)

    # EMA crossover signals
    df['ema_9']  = ta.trend.EMAIndicator(close, window=9).ema_indicator()
    df['ema_21'] = ta.trend.EMAIndicator(close, window=21).ema_indicator()
    df['ema_cross'] = df['ema_9'] - df['ema_21']  # positive = bullish crossover

    # ── VOLATILITY indicators ──────────────────────
    # Bollinger Bands: price channel showing overbought/oversold zones
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df['bb_upper']  = bb.bollinger_hband()
    df['bb_lower']  = bb.bollinger_lband()
    df['bb_mid']    = bb.bollinger_mavg()
    df['bb_pct']    = bb.bollinger_pband()   # 0=at lower band, 1=at upper band
    df['bb_width']  = bb.bollinger_wband()   # width of the band (volatility proxy)

    # ATR: Average True Range — how much TSLA moves per candle on average
    df['atr_14'] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()

    # ── MOMENTUM indicators ────────────────────────
    # Stochastic Oscillator: similar to RSI but compares to recent range
    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()

    # Rate of Change: % price change over N candles (raw momentum)
    df['roc_5']  = ta.momentum.ROCIndicator(close, window=5).roc()
    df['roc_10'] = ta.momentum.ROCIndicator(close, window=10).roc()

    # ── VOLUME indicators ──────────────────────────
    # Volume moving averages
    df['vol_sma_10'] = volume.rolling(10).mean()
    df['vol_ratio']  = volume / df['vol_sma_10']  # >1 = above-average volume spike

    # OBV: On-Balance Volume — cumulative volume pressure
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()

    # VWAP approximation per day (volume-weighted average price)
    df['typical_price'] = (high + low + close) / 3
    df['vwap_approx']   = (df['typical_price'] * volume).rolling(20).sum() / volume.rolling(20).sum()
    df['price_vs_vwap'] = (close - df['vwap_approx']) / df['vwap_approx'] * 100  # % above/below VWAP

    # ── TIME features (market microstructure) ─────
    # Time-of-day matters: open/close hours are more volatile
    df['hour']            = df.index.hour + df.index.minute / 60  # fractional hour
    df['is_first_30min']  = ((df.index.hour == 9)  & (df.index.minute < 60)).astype(int)
    df['is_last_30min']   = ((df.index.hour == 15) & (df.index.minute >= 30)).astype(int)
    df['is_power_hour']   = (df.index.hour == 15).astype(int)  # 3-4 PM is volatile

    # ── PRICE CONTEXT ─────────────────────────────
    # Where is current price relative to recent high/low?
    df['price_vs_high_20'] = (close - high.rolling(20).max()) / close * 100
    df['price_vs_low_20']  = (close - low.rolling(20).min())  / close * 100

    # Recent candle body size (bullish or bearish candle?)
    df['candle_body']      = (df['Close'] - df['Open']) / df['Open'] * 100

    # ── LAGGED FEATURES (previous candle context) ──
    for lag in [1, 2, 3]:
        df[f'close_lag_{lag}'] = close.shift(lag)
        df[f'rsi_lag_{lag}']   = df['rsi_14'].shift(lag)

    feature_cols = [c for c in df.columns if c not in
                    ['Open','High','Low','Close','Volume','future_close',
                     'future_pct_change','target','typical_price']]
    print(f"  Total features computed: {len(feature_cols)}")
    print(f"  Feature list:")
    for i, f in enumerate(feature_cols):
        print(f"    {i+1:>2}. {f}")

    return df


# ─────────────────────────────────────────────
# STEP 3: CLEAN UP
# ─────────────────────────────────────────────
def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)

    # Drop flat/ambiguous rows (target is NaN)
    df = df.dropna(subset=['target'])

    # Drop rows where indicators haven't warmed up yet (NaN values in features)
    df = df.dropna()

    after = len(df)
    print(f"\n  Rows before cleanup : {before:,}")
    print(f"  Rows after  cleanup : {after:,}  ({before - after} dropped — indicator warmup + flat rows)")
    return df


# ─────────────────────────────────────────────
# STEP 4: CHRONOLOGICAL TRAIN/VAL/TEST SPLIT
# ─────────────────────────────────────────────
def split_data(df: pd.DataFrame):
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

    train = df.iloc[:train_end]
    val   = df.iloc[train_end:val_end]
    test  = df.iloc[val_end:]

    print(f"\n" + "═"*58)
    print(f"  CHRONOLOGICAL DATA SPLIT")
    print(f"═"*58)
    print(f"  TRAIN set : {len(train):,} rows  ({train.index.min().date()} → {train.index.max().date()})")
    print(f"  VAL   set : {len(val):,} rows  ({val.index.min().date()} → {val.index.max().date()})")
    print(f"  TEST  set : {len(test):,} rows  ({test.index.min().date()} → {test.index.max().date()})")
    print(f"\n  ⚠  CRITICAL RULE: The TEST set is LOCKED until final evaluation.")
    print(f"     Never train or tune parameters using test set rows!")

    return train, val, test


# ─────────────────────────────────────────────
# STEP 5: BASELINE ACCURACY
# ─────────────────────────────────────────────
def compute_baselines(train, val):
    print(f"\n" + "═"*58)
    print(f"  BASELINE ACCURACY (the number your model must BEAT)")
    print(f"═"*58)

    # Majority class in training set
    majority_class = int(train['target'].mode()[0])
    majority_label = "UP" if majority_class == 1 else "DOWN"

    # If we always predict the majority class, what accuracy do we get?
    val_majority_acc = (val['target'] == majority_class).mean() * 100
    train_balance    = train['target'].mean() * 100  # % of UP labels

    print(f"\n  Training set class balance:")
    print(f"    UP   (1) : {train_balance:.1f}%")
    print(f"    DOWN (0) : {100 - train_balance:.1f}%")
    print(f"    Majority : {majority_label}")

    print(f"\n  Baseline 1 — Always predict '{majority_label}':")
    print(f"    Val accuracy: {val_majority_acc:.1f}%")

    print(f"\n  Baseline 2 — Random guess (50/50):")
    print(f"    Val accuracy: ~50.0%")

    target_accuracy = max(val_majority_acc, 50.0) + 5.0
    print(f"\n  ➡  Your model MUST achieve at least: {target_accuracy:.1f}% on the validation set")
    print(f"     (baseline + 5% minimum meaningful improvement)")
    print(f"\n  ➡  With confidence filtering (only predict when P>65% or P<35%):")
    print(f"     Target accuracy on confident signals: 60%+ is excellent for TSLA")

    return target_accuracy


# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────
def save_features(df: pd.DataFrame):
    os.makedirs("data", exist_ok=True)
    df.to_csv(OUT_PATH)
    print(f"\n  Features saved to: {OUT_PATH}")
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 58)
    print("  TSLA AI INDICATOR — Phase 1: Features & Target")
    print("=" * 58)

    df = load_data()

    df = add_target(df)
    df = add_features(df)
    df = clean_features(df)

    train, val, test = split_data(df)
    target_acc = compute_baselines(train, val)

    save_features(df)

    print(f"\n" + "═"*58)
    print(f"  Phase 1 Complete!")
    print(f"  Your ML-ready dataset is at: {OUT_PATH}")
    print(f"  The number to beat: {target_acc:.1f}% validation accuracy")
    print(f"  Next: Run src/step5_train_model.py")
    print("=" * 58)
