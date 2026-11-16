# TSLA AI INDICATOR — ACADEMIC PROJECT POSTER
## End-to-End Intraday Machine Learning Indicator for Tesla Stock

> **Institution**: IMPACT Lab, Department of Computer Science & Engineering, IIT Patna  
> **Student**: Vennela Jangiti (B.Tech CSE) | **Asset**: Tesla, Inc. (`NASDAQ: TSLA`)  
> **Horizon**: 10-Minute Directional Move ($\pm 0.08\%$) on 5-Minute Intraday Bars  

---

### Poster Dimensions & Suggested Layout
- **Orientation**: Landscape (48" $\times$ 36" standard conference poster or 16:9 widescreen slide for Canva / PowerPoint)
- **Visual Grid**: 3-Column Balanced Structure:
  - **Left Column**: Problem Context, Motivation, Data Pipeline, & 35 Stationary Features
  - **Center Column**: Core Machine Learning Architecture, Zero-Leakage Purging, & Live Dashboard Screenshot
  - **Right Column**: Academic Benchmark Results ($p=0.0362$), Cost-Aware Backtest (+6.01%), & Conclusions

---

```
+---------------------------------------------------------------------------------------------------------+
|                                    TITLE BANNER (Navy & Charcoal Theme)                                 |
|         AI-POWERED INTRADAY TRADING INDICATOR FOR TESLA USING MACHINE LEARNING                          |
|         Vennela Jangiti | IMPACT Lab, Dept. of Computer Science & Engineering, IIT Patna                |
+------------------------------------+------------------------------------+-------------------------------+
|       COLUMN 1: THE FOUNDATION     |       COLUMN 2: ARCHITECTURE       |      COLUMN 3: EMPIRICAL PROOF |
|                                    |                                    |                               |
| 1. PROBLEM & MOTIVATION            | 4. ML ARCHITECTURE & PURGING       | 6. ACADEMIC BENCHMARK RESULTS |
| • Intraday markets are noisy & non-| • Model: XGBoost Classifier        | • Held-Out Test Set (666 bars)|
|   stationary; traditional technical| • Platt Scaling: Probability       | • Confident Signals: N = 409  |
|   indicators use static thresholds |   calibration on validation set    | • Accuracy: 55.01%            |
|   without probabilistic confidence.| • Purging Protocol: 2-candle gap   | • 95% Wilson CI: [50.2%, 59.8%]|
| • Target: Directional move >=+0.08%|   between Train, Val, & Test sets  | • Precision UP: 53.3%         |
|   or <=-0.08% over next 10 minutes |   eliminating label overlap leakage| • Precision DOWN: 57.9%       |
|   (2 candles) on TSLA 5m bars.     | • Selective Filtering:             | • Binomial Test: p = 0.0362   |
|                                    |   BUY >= 55%, SELL <= 45%, HOLD    |   (Statistically Significant) |
| 2. DATA INGESTION & PIPELINE       |                                    |                               |
| • Provider: Alpaca Markets API     | 5. LIVE TRADING DASHBOARD (UI)     | 7. COST-AWARE BACKTEST (40 M) |
| • Feed: IEX (Same feed in training | [ INSERT DASHBOARD SCREENSHOT ]    | • Capital: $100,000           |
|   and live streaming = zero skew)  | • TradingView Lightweight Charts   | • Friction: 2.0 bps round-trip|
| • 120 Days / 83 Sessions / 4,516   | • Real-time WebSocket streaming    | • Strategy Return: +6.01%     |
|   clean 5-minute bars.             | • Heuristic ATR Risk Zones:        | • Buy & Hold TSLA: -6.10%     |
|                                    |   Base (0.8x), Bull (1.5x),        | • Net Alpha: +12.11%          |
| 3. 35 STATIONARY FEATURES          |   Bear Invalidation Stop (1.2x)    | • Sharpe Ratio: 4.72          |
| • Trend: RSI, MACD %, EMA Spreads  | • Immutable Prediction Ledger:     | • Max Drawdown: -7.63%        |
| • Volatility: Bollinger %b, ATR %  |   Closed-candle execution with     |   (vs -18.03% Buy & Hold)     |
| • Volume: Vol Ratio, OBV, VWAP Dist|   zero repainting guarantee        |                               |
| • Geometry: Shadows, Bodies, Lags  |                                    | 8. KEY CONCLUSIONS            |
| • ADF Test: All p < 0.001 (Zero raw|                                    | • Volume/VWAP features improve|
|   dollar price dependence).        |                                    |   signal quality and coverage.|
+------------------------------------+------------------------------------+-------------------------------+
```

---

### Detailed Section-by-Section Poster Content (Ready to Paste)

#### 1. Header & Metadata
- **Project Title**: AI-Powered Intraday Trading Indicator for Tesla Using Machine Learning
- **Subtitle**: Non-Repainting 10-Minute Directional Forecasting on 5-Minute Intraday Bars
- **Author**: Vennela Jangiti (Undergraduate Researcher)
- **Institution**: IMPACT Lab, Department of Computer Science and Engineering, Indian Institute of Technology Patna (IIT Patna)

#### 2. Left Column: Motivation, Data & Features
- **The Core Problem**: Classical technical indicators (RSI, MACD) are uncalibrated, evaluate single dimensions in isolation, and suffer from high false-positive rates during market consolidations.
- **Why Tesla (TSLA)?**: Extreme dollar liquidity (tight penny spreads), high natural intraday volatility (ATR $1.00–$3.50), and active institutional/retail momentum dynamics.
- **The Target Variable**: 2-candle forward return ($\pm 0.08\%$ threshold):
  - $UP \ge +0.08\%$ (Class 1)
  - $DOWN \le -0.08\%$ (Class 0)
  - Ambiguous chop ($\pm 0.08\%$) dropped during training to prevent noise fitting.
- **Stationary Feature Engineering**: All 35 predictors converted to percentages, oscillators, and ratios:
  - Augmented Dickey-Fuller (ADF) tests: $p < 0.001$ across all features.
  - Zero raw dollar price dependence prevents out-of-range tree failure.

#### 3. Center Column: Machine Learning Architecture & Dashboard
- **XGBoost Classifier**: 300 shallow trees (`max_depth = 3`), `learning_rate = 0.03`, regularized with L1 (`reg_alpha = 0.5`) and L2 (`reg_lambda = 2.0`).
- **Zero Data Leakage Protocol**:
  - Strict chronological splitting (70% Train / 15% Val / 15% Test).
  - **2-Candle Purging Buffer**: Enforces a 2-bar gap between splits, mathematically preventing 10-minute forward label overlap.
  - Backward-looking rolling features only.
- **Probability Calibration**: Fitted Platt logistic regression on validation probabilities, ensuring predicted confidence reflects true empirical win rates.
- **Live UI Engine**:
  - FastAPI backend polling Alpaca on closed candles every 60s.
  - TradingView Lightweight Charts with live candlestick and volume rendering.
  - ATR-based heuristic scenario zones (Base, Bull expansion, Bear stop-loss).
  - Immutable prediction ledger tracking all live calls with zero repainting.

#### 4. Right Column: Empirical Results & Institutional Backtesting
- **Statistically Significant Academic Edge**:
  - **Test Accuracy**: **55.01%** on $N = 409$ confident signals (out of 666 bars).
  - **Exact One-Sided Binomial Test**: $p = \mathbf{0.0362} < 0.05$ against the 50.5% majority baseline.
  - **Wilson 95% Confidence Interval**: $[50.2\%, 59.8\%]$.
  - **UP Precision**: **53.3%** (137/257) | **DOWN Precision**: **57.9%** (88/152).
- **Ablation Study**:
  - Indicators-Only (17 features): 54.81% accuracy ($N=405$).
  - Full Features (35 features): **55.01% accuracy ($N=409$)**, confirming volume and volatility features improve signal quality and throughput.
- **Realistic Cost-Aware Backtest (40 Marks Fulfill)**:
  - Evaluated on locked test set with **2.0 bps round-trip friction**.
  - **Net Strategy Return**: **+6.01%** (\$106,012.13 vs \$100,000 initial).
  - **Buy & Hold TSLA**: **-6.10%** (\$93,898.07).
  - **Net Alpha**: **+12.11% Outperformance**.
  - **Sharpe Ratio**: **4.72** | **Max Drawdown**: **-7.63%** (vs -18.03% Buy & Hold).
  - **Net Win Rate**: **55.02%** (230 Wins / 188 Losses).
