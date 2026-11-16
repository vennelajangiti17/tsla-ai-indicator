# AI-Powered Intraday Trading Indicator for Tesla (NASDAQ: TSLA)
## B.Tech CSE Project Poster Presentation

### Poster Metadata
- Title: AI-Powered Intraday Trading Indicator for Tesla (NASDAQ: TSLA) Using Purged Machine Learning
- Author: Jangiti Vennela
- Roll Number: 2401CS27
- Email: jangiti_2401cs27@iitp.ac.in
- Affiliation: Department of Computer Science and Engineering, Indian Institute of Technology Patna

---

### Slide Layout & Shape Mapping (Mentor's Sample Poster Structure)

#### 1. Header Bar
- Top Left: Indian Institute of Technology Patna Official Logo
- Top Right: Clean whitespace (IMPACT Lab logo removed as requested)
- Center Text Box (Clean hierarchy with reduced font sizes):
  - Line 1 (17pt bold): AI-Powered Intraday Trading Indicator for Tesla (NASDAQ: TSLA) Using Purged Machine Learning
  - Line 2 (12.5pt normal): Jangiti Vennela | Roll No: 2401CS27 | jangiti_2401cs27@iitp.ac.in
  - Line 3 (10pt normal): Department of Computer Science and Engineering, Indian Institute of Technology Patna

---

#### 2. Left Column (Independent non-overlapping bounds)

**Abstract (y = 1.12" to 2.83")**
This work develops an AI-powered intraday trading indicator for Tesla (NASDAQ: TSLA) on 5-minute candles predicting 10-minute directional price moves (±0.08% threshold). Using a purged time-series split and 17 technical features, an optimized Gradient Boosting classifier achieves 55.01% test accuracy (p = 0.0362) with 53.3% UP and 57.9% DOWN precision. In a cost-aware backtest with 2 bps friction, it produces +6.01% net return (Sharpe 4.72). Live NYSE forward-testing demonstrated a 66.7% win rate (10W / 5L) with zero repainting.

**Introduction (y = 2.95" to 5.18")**
- Intraday equities exhibit extreme noise and microstructural friction, causing retail indicators to lag severely [1].
- Standard cross-validation creates severe lookahead bias in non-stationary price series, leading to false historical performance [2].
- This project implements purged time-series splits, dead-zone filtering, and real-time WebSocket inference on live NYSE feeds.

**Problem Statement (y = 5.28" to 6.51")**
Predicting 10-minute directional price movements in volatile equities like TSLA requires overcoming market noise, spread friction, and lookahead bias without curve-fitting or indicator repainting.

---

#### 3. Center Column

**Key Contributions (y = 1.10" to 2.31")**
- Engineered a leak-free 17-feature pipeline (momentum, volatility, trend, volume) with rolling z-score scaling fit strictly on training splits.
- Designed a 2-step (10-minute) lookahead horizon with a ±0.08% dead-zone filter to remove 2,250 uninformative flat chop candles.
- Achieved 55.01% test accuracy (p = 0.0362) and +6.01% cost-aware net return; validated live on NYSE with a 66.7% win rate and zero repainting.

**Methodology (y = 2.38" to 4.76")**
- Figure 1: Full-color 4-stage architecture diagram (`assets/architecture_diagram.png`) with colored box fills (Mint, Blue, Purple, Amber), solid header banners, and small crisp typography.
- Caption: Figure 1: End-to-End System Architecture of TSLA AI Trading Indicator

**Algorithm 1: TSLA AI Intraday Directional Forecasting (y = 4.80" to 6.64")**
- Require: 5-minute OHLCV candles D; threshold delta = 0.08%; lookahead k = 2; cost c = 0.02%
- Ensure: Trained model m*, Directional signal S in {UP, DOWN}, Net equity curve
- 1. Compute 17 causal technical features (RSI, MACD, Bollinger Bands, ATR, EMA crosses).
- 2. Label y = UP if return > delta, DOWN if return < -delta; purge flat chop (|return| <= delta).
- 3. Split data chronologically (70% Train, 15% Val, 15% Test) with 2-bar purge buffers.
- 4. Standardize using Train parameters; train Logistic Regression, Random Forest, GBDT.
- 5. In real-time inference on closed 5-min candle t:
     - Predict P(UP), P(DOWN); apply confidence filter (|P - 0.50| >= 0.05).
     - Deduct 2.0 bps round-trip friction and update cumulative strategy equity.
- 6. Return optimal model m*, out-of-sample metrics (55.01% Acc, Sharpe 4.72), and live signal.

---

#### 4. Right Column

**Result and Discussion (y = 1.10" to 5.55")**
- Table 1 Caption: Table 1: Proposed Model Results Compared to Baselines (N = 409)
- Table 1 Image (`assets/table1_results.png`):
  - Baseline (Always UP): Test Acc 50.45%, Prec UP 50.45%, Prec DN 0.00%, p-value 1.0000, Net Return -6.10%
  - RSI Heuristic Rule: Test Acc 50.45%, Prec UP 50.52%, Prec DN 50.00%, p-value 0.5155, Net Return -4.22%
  - Logistic Regression: Test Acc 51.05%, Prec UP 51.32%, Prec DN 50.70%, p-value 0.3932, Net Return -1.15%
  - Random Forest: Test Acc 52.81%, Prec UP 52.10%, Prec DN 53.60%, p-value 0.1240, Net Return +1.84%
  - Gradient Boosting (Ours): Test Acc 55.01%, Prec UP 53.31%, Prec DN 57.89%, p-value 0.0362, Net Return +6.01%
- Figure 2 Image (`assets/results_plots.png`):
  - (a) Cost-Aware Out-of-Sample Backtest (+6.01% vs -6.10% TSLA Buy & Hold, 2 bps friction)
  - (b) Real-Time Live Forward-Test on NYSE Feeds (66.7% Win Rate on 15 Resolved Trades)

**Conclusion (y = 5.60" to 6.62")**
The Gradient Boosting framework demonstrates an authentic, statistically validated intraday edge (55.01% test accuracy, p = 0.0362, Sharpe 4.72). Live NYSE testing confirms operational stability with a 66.7% win rate, validating that temporal purging and dead-zone filtering produce robust trading signals.

---

#### 5. Footer Bar (Exact references from ML Project Report PDF)

**References:**
[1] Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 785–794.
[2] Lopez de Prado, M. (2018). Advances in Financial Machine Learning. John Wiley & Sons, Hoboken, New Jersey.
[3] Fischer, T., & Krauss, C. (2018). Deep learning with long short-term memory networks for financial market predictions. European Journal of Operational Research, 270(2), 654–669.
