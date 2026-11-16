# Dataset Audit & Data Dictionary (TSLA Intraday 5-Min)
*Generated according to IMPACT Lab ML Guidelines (Step 3 & Step 6)*

## 1. Dataset Card
- **Asset**: Tesla, Inc. (NASDAQ: TSLA)
- **Source**: Alpaca Markets Historical Data API (IEX Feed)
- **Timeframe**: 5-Minute OHLCV bars
- **Date Range**: 2026-07-13 to 2026-11-09 (approx. 120 calendar days)
- **Total Valid Observations**: 4,880
- **Missing Values**: 0
- **Duplicate Timestamps**: 0
- **Target Definition**: Binary direction after 2 candles (10 minutes) with threshold $\pm 0.08\%$
  - Class 1 (UP): 2,469 (50.59%)
  - Class 0 (DOWN): 2,411 (49.41%)
  - Balance Ratio: 1.024 (Well-balanced)

## 2. Feature Dictionary (35 Features)
| # | Feature Name | Category | Description |
|---|--------------|----------|-------------|
| 1 | `rsi_14` | Momentum | 14-period Relative Strength Index (0-100) |
| 2 | `macd` | Trend | Moving Average Convergence Divergence line |
| 3 | `macd_signal` | Trend | 9-period EMA of MACD line |
| 4 | `macd_hist` | Trend | Difference between MACD line and Signal line |
| 5 | `ema_9` | Trend | 9-period Exponential Moving Average |
| 6 | `ema_21` | Trend | 21-period Exponential Moving Average |
| 7 | `ema_cross` | Trend | `ema_9 - ema_21` (Fast EMA minus slow EMA) |
| 8 | `bb_upper` | Volatility | Bollinger Band upper boundary (20 periods, 2 std) |
| 9 | `bb_lower` | Volatility | Bollinger Band lower boundary (20 periods, 2 std) |
| 10 | `bb_mid` | Volatility | Bollinger Band middle moving average (20 periods) |
| 11 | `bb_pct` | Volatility | %B oscillator: location within Bollinger Bands |
| 12 | `bb_width` | Volatility | Bandwidth proxy for volatility expansion/contraction |
| 13 | `atr_14` | Volatility | 14-period Average True Range in dollars |
| 14 | `stoch_k` | Momentum | Fast stochastic oscillator %K |
| 15 | `stoch_d` | Momentum | Slow stochastic oscillator %D (3-period smoothed %K) |
| 16 | `roc_5` | Momentum | 5-period Rate of Change percentage |
| 17 | `roc_10` | Momentum | 10-period Rate of Change percentage |
| 18 | `vol_sma_10` | Volume | 10-period Simple Moving Average of Volume |
| 19 | `vol_ratio` | Volume | Current candle volume divided by `vol_sma_10` |
| 20 | `obv` | Volume | On-Balance Volume cumulative metric |
| 21 | `vwap_approx` | Volume-Price | 20-period rolling Volume-Weighted Average Price |
| 22 | `price_vs_vwap` | Volume-Price | % deviation of Close price from VWAP |
| 23 | `hour` | Time/Microstructure | Fractional hour of day (e.g., 9.5 for 9:30 AM) |
| 24 | `is_first_30min` | Time/Microstructure | Binary flag: 1 if regular market opening 30 min (9:30-10:00 EST) |
| 25 | `is_last_30min` | Time/Microstructure | Binary flag: 1 if closing 30 min (15:30-16:00 EST) |
| 26 | `is_power_hour` | Time/Microstructure | Binary flag: 1 if during 15:00-16:00 EST |
| 27 | `price_vs_high_20` | Context | % distance from 20-candle rolling high |
| 28 | `price_vs_low_20` | Context | % distance from 20-candle rolling low |
| 29 | `candle_body` | Price Action | Close-Open percentage return of current candle |
| 30 | `close_lag_1` | Autoregressive | Close price 1 candle ago |
| 31 | `rsi_lag_1` | Autoregressive | RSI 1 candle ago |
| 32 | `close_lag_2` | Autoregressive | Close price 2 candles ago |
| 33 | `rsi_lag_2` | Autoregressive | RSI 2 candles ago |
| 34 | `close_lag_3` | Autoregressive | Close price 3 candles ago |
| 35 | `rsi_lag_3` | Autoregressive | RSI 3 candles ago |

## 3. Data Leakage Prevention Protocol
1. **Chronological Splitting**: Split strictly by time. Training: first 70%, Validation: middle 15%, Test: final 15% (Held-out).
2. **Feature Transformers**: Any normalization or scaling (e.g., for Logistic Regression) is fitted **strictly on the training fold** and transformed onto validation and test sets.
3. **Target Calculation**: Target looks forward 2 candles into future. All predictive features use only data available at or before candle close.
