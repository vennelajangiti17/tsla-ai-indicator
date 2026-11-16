# TSLA AI Indicator — Quantitative Decision Log & Experiment Notes

> **Institution**: IMPACT Lab, Department of Computer Science & Engineering, IIT Patna  
> **Coursework Project**: End-to-End Machine Learning Indicator  
> **Asset**: Tesla, Inc. (`NASDAQ: TSLA`) | **Timeframe**: 5-Minute Intraday Bars  

---

## Phase 0 — Setup & Data Audit

### Data Source
- **Provider**: Alpaca Markets (Paper Trading API)
- **Feed**: IEX (Investors Exchange)
- **Why Alpaca**: Provides unified programmatic access to both historical 5-minute bars and real-time live polling.
- **Why IEX is valid**: For highly liquid equities like TSLA, IEX prices track the consolidated SIP tape within cents. Training and live inference use the exact same feed, guaranteeing zero data distribution skew.

### Dataset Specifications

| Field | Value |
|---|---|
| **Ticker** | TSLA |
| **Candle Timeframe** | 5-Minute OHLCV |
| **Observation Window** | 2026-07-13 to 2026-11-06 (approx. 120 calendar days) |
| **Total Raw Candles** | 6,904 bars |
| **Clean Preprocessed Rows** | 4,516 candles (after indicator warmup & flat filtering) |
| **Trading Days** | 83 trading sessions |
| **Avg Candles / Day** | ~78 bars (full 6.5-hour regular session) |
| **Missing Values** | 0 (verified zero NaNs in clean dataset) |
| **Timezone** | `America/New_York` (EST/EDT) |
| **IST Market Window** | 8:00 PM – 1:30 AM IST (regular US market session) |

---

## Phase 1 — Target Formulation & Leakage Prevention

- **Prediction Horizon**: Exactly **2 candles ahead (10 minutes)**.
- **Session Boundary Protection**: Shift operations are grouped strictly by trading session (`groupby(df.index.date)`). End-of-day candles never shift into the next morning's market open, eliminating overnight gap distortion.
- **Continuous Horizon Enforcement**: Verified time difference is strictly 10.0 minutes (dropping session halts/gaps).
- **Directional Threshold**: $\pm 0.08\%$ minimum return threshold.
  - Returns $> +0.08\% \rightarrow$ Class 1 (UP)
  - Returns $< -0.08\% \rightarrow$ Class 0 (DOWN)
  - Returns between $-0.08\%$ and $+0.08\% \rightarrow$ Dropped as ambiguous market chop.
- **Class Balance**: 
  - Class 1 (UP): 2,278 (50.4%)
  - Class 0 (DOWN): 2,238 (49.6%)
- **Data Splitting**: Strictly chronological split (NO random shuffling to prevent future-to-past data leakage):
  - **Train**: First 70% (3,161 candles, July 13 to Oct 2)
  - **Validation**: Next 15% (677 candles, Oct 2 to Oct 22)
  - **Held-Out Test**: Final 15% (678 candles, Oct 22 to Nov 6) — **Locked until final single-pass evaluation**.

---

## Phase 2 — Model Architecture & Walk-Forward Validation

- **Model**: **XGBoost Classifier (`XGBClassifier`)**
- **Hyperparameters**:
  - `n_estimators = 300`
  - `max_depth = 3` (limits tree complexity to avoid memorizing high-frequency noise)
  - `learning_rate = 0.03`
  - `subsample = 0.8`, `colsample_bytree = 0.8`
  - `min_child_weight = 15`, `gamma = 0.5`, `reg_alpha = 0.5`, `reg_lambda = 2.0`
- **Stationary Feature Set**: 35 normalized features (zero raw dollar levels used):
  - Trend: `rsi_14`, `macd_pct`, `macd_signal_pct`, `macd_hist_pct`, `price_vs_ema_9`, `price_vs_ema_21`, `ema_cross_pct`.
  - Volatility: `price_vs_bb_upper`, `price_vs_bb_lower`, `price_vs_bb_mid`, `bb_pct`, `bb_width`, `atr_pct`.
  - Momentum: `stoch_k`, `stoch_d`, `roc_5`, `roc_10`.
  - Volume: `vol_ratio`, `obv_zscore` (rolling 20-candle z-score), `price_vs_vwap`.
  - Time/Microstructure: `hour`, `is_first_30min`, `is_last_30min`, `is_power_hour`.
  - Context & Return Lags: `price_vs_high_20`, `price_vs_low_20`, `candle_body`, `upper_shadow`, `lower_shadow`, `ret_lag_1,2,3`, `rsi_lag_1,2`, `vol_ratio_lag_1`.

### 5-Fold Walk-Forward Cross-Validation Results
*Simulating rolling-origin trading over time:*
- Fold 1: Raw = 51.0%, Confident = 52.3%, UP Precision = 67.1%
- Fold 2: Raw = 48.7%, Confident = 48.9%, UP Precision = 54.3%
- Fold 3: Raw = 49.8%, Confident = 51.0%, UP Precision = 53.1%
- Fold 4: Raw = 51.3%, Confident = 51.3%, UP Precision = 49.5%
- Fold 5: Raw = 50.4%, Confident = 52.1%, UP Precision = 51.4%
- **Average Across 5 Folds**: Raw = **50.23%**, Confident = **51.12%**.

### Threshold Tuning & Final Performance

| Dataset | Raw Accuracy | Confident Accuracy ($P \ge 0.52$) | Confident Precision (UP) | Coverage |
|---|---|---|---|---|
| **Validation Set** | 53.2% | 54.2% | **54.7%** | 79.6% |
| **Held-Out Test Set** | 53.2% | 53.8% | **55.0%** | 85.0% |

- **Comparison to Baselines**:
  - Majority Baseline (Always UP): 51.3% test accuracy.
  - RSI Rule Baseline ($<35$ Buy, $>65$ Sell): 52.1% test accuracy.
  - Logistic Regression: 49.0% test accuracy.
  - **Proposed XGBoost**: **53.8% test accuracy / 55.0% precision on UP signals**.

---

## Phase 3 — Multi-Scenario Forecast Engine

- **Base Target**: Current Price $\pm$ (ATR $\times$ Conviction Multiplier). Conviction scales dynamically with model probability.
- **Bull Target**: Current Price $+$ (ATR $\times 1.5$). Extended to $1.3\times$ if Volume Ratio $> 2.0\times$ average.
- **Bear Target**: Current Price $-$ (ATR $\times 1.2$). Defines the explicit hard invalidation stop-loss zone.
- **Dynamic Context Generator**: Evaluates actual indicator state in real time:
  - If RSI $< 35$: flags oversold mean-reversion setup.
  - If RSI $> 65$: flags overbought momentum extension.
  - If Volume $> 2\times$: flags high volume breakout.

---

## Phase 4 & 5 — Production Backend & Live Dashboard

- **Server Stack**: FastAPI, Uvicorn, WebSockets.
- **Anti-Repainting Guarantee**: Evaluates strictly on **closed 5-minute bars**. Live signals are immutable once logged.
- **True 10-Minute Queue**: Each prediction is held in a 2-candle pending queue (`candles_passed < 2`). Upon the close of the 2nd candle (10 minutes), the price move is measured against the $\pm 0.08\%$ threshold and assigned ✅ (Win), ❌ (Loss), or 🟡 (Flat).
- **Disk Persistence**: Predictions are automatically saved to `logs/prediction_log.json` so server restarts or browser refreshes never wipe live testing history.
- **Frontend**: TradingView Lightweight Charts (v4.1.3), live candlesticks, buy/sell markers, indicator gauges, NYSE market status badge, and real-time prediction log table.
