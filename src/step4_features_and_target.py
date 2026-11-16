"""
=============================================================
Phase 1 | Step 4: Stationary Feature Engineering & Target
=============================================================
Major Improvements Addressing Rigorous Quant Review:
  1. SESSION-AWARE TARGET:
     - Grouped by trading date (NEVER crosses overnight boundaries).
     - Verified exact 10.0-minute horizon (drops gaps/halts).
     - Move threshold: +/- 0.08% (drops flat noise).
  2. 100% STATIONARY FEATURES:
     - No raw price levels (no raw Close, EMA, BB dollar levels).
     - All prices converted to percentage returns, distances (% vs EMA, % vs BB).
     - OBV converted to rolling 20-candle z-score (identical scale in live buffer vs train).
     - Volatility normalized as % of price (ATR%).
  3. DATA INTEGRITY:
     - Strict chronological ordering.
=============================================================
"""

import os, sys
import numpy as np
import pandas as pd
import ta

CSV_PATH = "data/tsla_5m_historical.csv"
OUT_PATH = "data/tsla_features.csv"

LOOKAHEAD_CANDLES = 2    # 2 candles = exactly 10 minutes
MIN_MOVE_PCT      = 0.08 # +/- 0.08% move threshold
TRAIN_RATIO       = 0.70
VAL_RATIO         = 0.15

def load_data():
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: '{CSV_PATH}' not found. Run step1 first.")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("America/New_York")
    df = df.sort_index()
    print(f"  Loaded {len(df):,} candles from {CSV_PATH}")
    return df

def add_target(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n  Defining session-aware target: exactly 10 minutes ({LOOKAHEAD_CANDLES} candles) ahead...")
    
    # 1. Group by calendar date: shift(-2) ONLY within the same day
    df['future_close'] = df.groupby(df.index.date)['Close'].shift(-LOOKAHEAD_CANDLES)
    df['future_time']  = df.groupby(df.index.date)['Close'].transform(
        lambda s: s.index.to_series().shift(-LOOKAHEAD_CANDLES)
    )
    
    # 2. Verify exact 10-minute horizon (no missing candles / session gaps)
    time_diff_min = (df['future_time'] - df.index.to_series()).dt.total_seconds() / 60.0
    is_exact_10m = (time_diff_min == 10.0)
    
    # 3. Percentage change
    df['future_pct_change'] = (df['future_close'] - df['Close']) / df['Close'] * 100.0
    
    # 4. Target assignment: 1 = UP (> +0.08%), 0 = DOWN (< -0.08%), NaN = flat or session-end
    df['target'] = np.nan
    valid_mask = is_exact_10m & df['future_close'].notna()
    
    df.loc[valid_mask & (df['future_pct_change'] >  MIN_MOVE_PCT), 'target'] = 1
    df.loc[valid_mask & (df['future_pct_change'] < -MIN_MOVE_PCT), 'target'] = 0
    # All other rows stay NaN and will be dropped in clean_features

    up_count   = (df['target'] == 1).sum()
    down_count = (df['target'] == 0).sum()
    dropped    = df['target'].isna().sum()

    print(f"    UP   (1) : {up_count:,} ({up_count/(up_count+down_count)*100:.1f}%)")
    print(f"    DOWN (0) : {down_count:,} ({down_count/(up_count+down_count)*100:.1f}%)")
    print(f"    Flat / Gap / End-of-Day (Dropped): {dropped:,}")

    return df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n  Computing 35 strictly stationary technical features...")
    close  = df['Close']
    high   = df['High']
    low    = df['Low']
    open_p = df['Open']
    volume = df['Volume']

    # ── 1. TREND (Stationary % Distances & Oscillators) ──
    df['rsi_14'] = ta.momentum.RSIIndicator(close, window=14).rsi()
    
    macd_obj = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df['macd_pct']        = (macd_obj.macd() / close) * 100.0
    df['macd_signal_pct'] = (macd_obj.macd_signal() / close) * 100.0
    df['macd_hist_pct']   = (macd_obj.macd_diff() / close) * 100.0

    ema_9  = ta.trend.EMAIndicator(close, window=9).ema_indicator()
    ema_21 = ta.trend.EMAIndicator(close, window=21).ema_indicator()
    df['price_vs_ema_9']  = (close - ema_9) / ema_9 * 100.0
    df['price_vs_ema_21'] = (close - ema_21) / ema_21 * 100.0
    df['ema_cross_pct']   = (ema_9 - ema_21) / ema_21 * 100.0

    # ── 2. VOLATILITY (Normalized) ──
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    bb_upper = bb.bollinger_hband()
    bb_lower = bb.bollinger_lband()
    bb_mid   = bb.bollinger_mavg()
    df['price_vs_bb_upper'] = (close - bb_upper) / close * 100.0
    df['price_vs_bb_lower'] = (close - bb_lower) / close * 100.0
    df['price_vs_bb_mid']   = (close - bb_mid) / bb_mid * 100.0
    df['bb_pct']            = bb.bollinger_pband()
    df['bb_width']          = bb.bollinger_wband()
    
    atr_14 = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    df['atr_pct'] = (atr_14 / close) * 100.0   # ATR as % of current price

    # ── 3. MOMENTUM ──
    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    df['roc_5']   = ta.momentum.ROCIndicator(close, window=5).roc()
    df['roc_10']  = ta.momentum.ROCIndicator(close, window=10).roc()

    # ── 4. VOLUME & LIQUIDITY (Rolling & Scale-Free) ──
    vol_sma_10      = volume.rolling(10).mean()
    df['vol_ratio'] = volume / (vol_sma_10 + 1e-8)
    
    # Rolling Z-score of OBV over 20 candles (prevents infinite accumulation divergence)
    raw_obv = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    obv_mean = raw_obv.rolling(20).mean()
    obv_std  = raw_obv.rolling(20).std() + 1e-8
    df['obv_zscore'] = (raw_obv - obv_mean) / obv_std

    typical_price = (high + low + close) / 3.0
    vwap_approx   = (typical_price * volume).rolling(20).sum() / (volume.rolling(20).sum() + 1e-8)
    df['price_vs_vwap'] = (close - vwap_approx) / vwap_approx * 100.0

    # ── 5. TIME & MICROSTRUCTURE ──
    df['hour']           = df.index.hour + df.index.minute / 60.0
    df['is_first_30min'] = ((df.index.hour == 9)  & (df.index.minute >= 30)).astype(int)
    df['is_last_30min']  = ((df.index.hour == 15) & (df.index.minute >= 30)).astype(int)
    df['is_power_hour']  = (df.index.hour == 15).astype(int)

    # ── 6. PRICE ACTION PATTERNS & RETURNING LAGS ──
    df['price_vs_high_20'] = (close - high.rolling(20).max()) / close * 100.0
    df['price_vs_low_20']  = (close - low.rolling(20).min())  / close * 100.0
    df['candle_body']      = (close - open_p) / open_p * 100.0
    df['upper_shadow']     = (high - np.maximum(open_p, close)) / close * 100.0
    df['lower_shadow']     = (np.minimum(open_p, close) - low) / close * 100.0

    # Percentage returns for past candles (NOT raw prices!)
    df['ret_lag_1'] = close.pct_change(1) * 100.0
    df['ret_lag_2'] = close.pct_change(2) * 100.0
    df['ret_lag_3'] = close.pct_change(3) * 100.0
    df['rsi_lag_1'] = df['rsi_14'].shift(1)
    df['rsi_lag_2'] = df['rsi_14'].shift(2)
    df['vol_ratio_lag_1'] = df['vol_ratio'].shift(1)

    drop_non_features = [
        'Open','High','Low','Close','Volume',
        'future_close','future_time','future_pct_change','target'
    ]
    feature_cols = [c for c in df.columns if c not in drop_non_features]
    print(f"  Total stationary features: {len(feature_cols)}")
    return df

def clean_and_save(df: pd.DataFrame):
    before = len(df)
    # Drop rows where target is NaN (flats, gaps, end of days)
    df = df.dropna(subset=['target'])
    # Drop indicator warmup rows
    df = df.dropna()
    after = len(df)
    print(f"\n  Rows before cleanup : {before:,}")
    print(f"  Clean ML rows       : {after:,}  ({before - after} dropped)")
    
    df.to_csv(OUT_PATH)
    print(f"  Saved clean stationary dataset to: {OUT_PATH}")
    return df

if __name__ == "__main__":
    df = load_data()
    df = add_target(df)
    df = add_features(df)
    df = clean_and_save(df)
