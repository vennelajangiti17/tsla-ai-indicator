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


---

## Phase 6 — Purging, Calibration & Statistical Rigor (Audit Upgrade)

- **2-Candle Purging (`PURGE_BARS = 2`)**: To prevent 10-minute label overlap leakage between splits, a 2-candle gap was purged at the boundaries of Train-Val and Val-Test splits.
- **Statistical Significance (One-Sided Binomial Test)**:
  - Evaluation on Test Set: $N = 409$ confident signals ($P \ge 0.55$ or $P \le 0.45$).
  - Correct Calls: 225 out of 409 (**55.01% accuracy**).
  - Majority Class Baseline: 50.5% (336 / 666).
  - Exact Binomial Test $p$-value: **$p = 0.0362 < 0.05$** (Statistically significant at 95% confidence).
  - Wilson 95% Confidence Interval: **[50.2%, 59.8%]**.
  - UP Precision: **53.3%** (137 / 257).
  - DOWN Precision: **57.9%** (88 / 152).
- **Feature Ablation Study**:
  - Indicators-Only (17 classical features): 54.81% accuracy, 60.8% coverage.
  - Full Features (35 features including volume, VWAP, session timing): **55.01% accuracy, 61.4% coverage**.
  - Confirms volume and volatility features improve signal confidence and actionable throughput.
- **Platt Probability Calibration**:
  - Logistic regression scaler fitted strictly on validation set predictions.
  - Reliability diagram and distribution histogram generated at `results/calibration_curve.png`.

---

## Phase 7 — Cost-Aware Friction Backtesting (40 Marks Fulfill)

- **Execution Cost**: **2.0 basis points ($0.02\%$) round-trip** (1 bps spread + 1 bps slippage).
- **Test Period**: Oct 22 to Nov 06, 2026 (666 bars).
- **Results**:
  - Initial Capital: $100,000.00
  - Proposed Strategy Final Capital: **$106,012.13 (+6.01% Net Return)**
  - Buy & Hold TSLA Final Capital: **$93,898.07 (-6.10% Net Return)**
  - Net Strategy Alpha: **+12.11% Outperformance**
  - Annualized Sharpe Ratio: **4.72** (vs -0.42 Buy & Hold)
  - Maximum Drawdown: **-7.63%** (vs -18.03% Buy & Hold)
  - Net Win Rate: **55.02%** (230 Wins / 188 Losses)
  - Profit Factor: **1.10**
- **Artifacts Generated**: `results/equity_curve.png` and `results/cost_aware_backtest.md`.
