# Dataset Audit & Data Dictionary (TSLA Intraday 5-Min)
*Generated according to IMPACT Lab ML Guidelines (Step 3 & Step 6)*

## 1. Dataset Card
- **Asset**: Tesla, Inc. (`NASDAQ: TSLA`)
- **Data Source**: Alpaca Markets Historical Data API (IEX Feed)
- **Timeframe**: 5-Minute OHLCV candles
- **Observation Period**: 2026-07-13 to 2026-11-06
- **Total Valid Observations**: 4,516 clean candles
- **Missing Values**: 0
- **Duplicate Timestamps**: 0
- **Horizon & Target**: Exactly 10 minutes (2 candles) ahead, within same trading session.
  - Direction threshold: $\pm 0.08\%$
  - Class 1 (UP): 2,292 (50.75%)
  - Class 0 (DOWN): 2,224 (49.25%)
  - Balance: Well-balanced 50/50 distribution.

## 2. Stationary Feature Dictionary (35 Features)
All features are 100% stationary (percentage returns, distances, and oscillators). Zero raw dollar price levels are used.

| # | Feature Name | Category | Formula / Description |
|---|--------------|----------|-----------------------|
| 1 | `rsi_14` | Momentum | 14-period RSI (0-100) |
| 2 | `macd_pct` | Trend | (MACD / Close) * 100 |
| 3 | `macd_signal_pct` | Trend | (MACD Signal / Close) * 100 |
| 4 | `macd_hist_pct` | Trend | (MACD Histogram / Close) * 100 |
| 5 | `price_vs_ema_9` | Trend | % distance: (Close - EMA9) / EMA9 * 100 |
| 6 | `price_vs_ema_21` | Trend | % distance: (Close - EMA21) / EMA21 * 100 |
| 7 | `ema_cross_pct` | Trend | % difference: (EMA9 - EMA21) / EMA21 * 100 |
| 8 | `price_vs_bb_upper`| Volatility | % distance: (Close - BB Upper) / Close * 100 |
| 9 | `price_vs_bb_lower`| Volatility | % distance: (Close - BB Lower) / Close * 100 |
| 10 | `price_vs_bb_mid` | Volatility | % distance: (Close - BB Mid) / BB Mid * 100 |
| 11 | `bb_pct` | Volatility | Bollinger %B oscillator (0 to 1) |
| 12 | `bb_width` | Volatility | Bollinger Bandwidth: (Upper - Lower) / Mid |
| 13 | `atr_pct` | Volatility | 14-period ATR as % of Close price |
| 14 | `stoch_k` | Momentum | Fast Stochastic %K (0-100) |
| 15 | `stoch_d` | Momentum | Slow Stochastic %D (0-100) |
| 16 | `roc_5` | Momentum | 5-candle percentage return |
| 17 | `roc_10` | Momentum | 10-candle percentage return |
| 18 | `vol_ratio` | Volume | Volume / 10-period Volume SMA |
| 19 | `obv_zscore` | Volume | Rolling 20-candle Z-score of On-Balance Volume |
| 20 | `price_vs_vwap` | Volume-Price | % distance from 20-candle rolling VWAP |
| 21 | `hour` | Time/Micro | Fractional hour of day (e.g. 9.5 for 9:30 AM EST) |
| 22 | `is_first_30min` | Time/Micro | 1 if 9:30-10:00 AM EST, 0 otherwise |
| 23 | `is_last_30min` | Time/Micro | 1 if 15:30-16:00 EST, 0 otherwise |
| 24 | `is_power_hour` | Time/Micro | 1 if 15:00-16:00 EST, 0 otherwise |
| 25 | `price_vs_high_20`| Context | % distance from 20-candle rolling High |
| 26 | `price_vs_low_20` | Context | % distance from 20-candle rolling Low |
| 27 | `candle_body` | Price Action | % return of candle: (Close - Open) / Open * 100 |
| 28 | `upper_shadow` | Price Action | Upper wick size as % of Close |
| 29 | `lower_shadow` | Price Action | Lower wick size as % of Close |
| 30 | `ret_lag_1` | Autoregressive | 1-candle % return (Close.pct_change(1)) |
| 31 | `ret_lag_2` | Autoregressive | 2-candle % return (Close.pct_change(2)) |
| 32 | `ret_lag_3` | Autoregressive | 3-candle % return (Close.pct_change(3)) |
| 33 | `rsi_lag_1` | Autoregressive | RSI 1 candle ago |
| 34 | `rsi_lag_2` | Autoregressive | RSI 2 candles ago |
| 35 | `vol_ratio_lag_1` | Volume | Volume ratio 1 candle ago |

## 3. Data Leakage Prevention Protocol
1. **Chronological Splitting**: Chronological order strictly preserved (Train: 70%, Val: 15%, Test: 15%).
2. **Session Boundary Guard**: Target computation uses `groupby(df.index.date)`. Shifting never looks into the next day.
3. **Exact Time-Span Enforcement**: Horizon strictly required to be exactly 10.0 minutes. In-between gap candles are dropped.
