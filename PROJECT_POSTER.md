# AI-Powered Intraday Trading Indicator for Tesla (NASDAQ: TSLA)
## B.Tech CSE Project Poster Presentation

### Poster Metadata
- Title: AI-Powered Intraday Trading Indicator for Tesla (NASDAQ: TSLA) Using Purged Machine Learning
- Author: Jangiti Vennela
- Roll Number: 2401CS27
- Email: vennela_2401cs27@iitp.ac.in
- Affiliation: Department of Computer Science and Engineering, Indian Institute of Technology Patna
- Lab: IMPACT Lab, IIT Patna

---

### Slide Layout Mapping (Mentor's Sample Poster Structure)

#### 1. Header Bar
- Top Left: Indian Institute of Technology Patna Official Logo
- Center:
  - Line 1: AI-Powered Intraday Trading Indicator for Tesla (NASDAQ: TSLA) Using Purged Machine Learning
  - Line 2: Jangiti Vennela | Roll No: 2401CS27 | vennela_2401cs27@iitp.ac.in
  - Line 3: Department of Computer Science and Engineering, Indian Institute of Technology Patna
- Top Right: IMPACT Lab Official Logo

---

#### 2. Left Column

**Abstract**
This work presents an AI-powered intraday trading indicator for Tesla (NASDAQ: TSLA) on 5-minute candles to predict short-term 10-minute directional price movements (±0.08% threshold). Using a purged time-series split and 17 engineered technical features across trend, momentum, and volatility, an optimized Gradient Boosting classifier achieves 55.01% directional test accuracy (statistically significant, p = 0.0362) and a two-sided precision of 53.3% UP and 57.9% DOWN. In a cost-aware backtest with 2 bps friction, the strategy generates +6.01% net return versus -6.10% buy-and-hold (Sharpe 4.72). A live NYSE forward test verified real-time stability with a 66.7% win rate (10 wins / 5 losses) over 44 candles with zero repainting.

**Introduction**
- Intraday equities exhibit extreme noise and microstructural friction, making retail technical indicators prone to false breakouts and severe lag [1].
- Machine learning offers adaptive multi-feature pattern recognition, but standard cross-validation risks lookahead bias and train-test contamination in non-stationary price series [2].
- This study develops a leak-free, cost-aware ML indicator with automated purged time-series validation and real-time WebSocket deployment on live NYSE feeds.

**Problem Statement**
Predicting 10-minute directional price shifts in volatile equities like TSLA requires overcoming market noise, spread friction, and lookahead bias without curve-fitting or indicator repainting.

---

#### 3. Center Column

**Key Contributions**
- Engineered a leak-free 17-feature pipeline (momentum, volatility, trend, volume) with rolling z-score scaling fit strictly on training splits.
- Designed a 2-step (10-minute) lookahead horizon with a ±0.08% dead-zone filter to remove 2,250 uninformative flat chop candles.
- Achieved 55.01% test accuracy (p = 0.0362) and +6.01% cost-aware net return; validated live on NYSE with a 66.7% win rate and zero repainting.

**Methodology**
- [Image: Figure 1 Architecture Diagram - assets/architecture_diagram.png]
- Caption: Figure 1: End-to-End System Architecture of TSLA AI Trading Indicator

**Algorithm 1: TSLA AI Intraday Directional Forecasting**
- Require: 5-minute OHLCV candles D = {(t, O, H, L, C, V)}; threshold delta = 0.08%; lookahead k = 2; cost c = 0.02%
- Ensure: Optimal model m*, Directional signal S, Net equity curve
- 1. Compute 17 causal technical features (RSI, MACD, Bollinger Bands, ATR, EMA crosses).
- 2. Label y = UP if (C_{t+2} - C_t)/C_t > delta, DOWN if < -delta, else NEUTRAL (purged from training).
- 3. Chronological split: 70% Train (3,104 bars), 2-bar purge, 15% Val (664 bars), 2-bar purge, 15% Test (666 bars).
- 4. Fit StandardScaler on Train; train Logistic Regression, Random Forest, and Gradient Boosting.
- 5. In real-time inference on closed candle t:
     - Predict P(UP), P(DOWN); apply confidence filter (|P - 0.50| >= 0.05).
     - Simulate trade execution deducting 2.0 bps friction; track cumulative account equity.
- 6. Return optimal model m*, out-of-sample metrics (55.01% Acc, Sharpe 4.72), and live signal.

---

#### 4. Right Column

**Result and Discussion**
- Table Caption: Table 1: Proposed Model Results Compared to Baselines (N = 409)
- [Image: Table 1 Image - assets/table1_results.png]
  - Baseline (Always UP): Test Acc 50.45%, Prec UP 50.45%, Prec DN 0.00%, p-value 1.0000, Net Return -6.10%
  - RSI Heuristic Rule: Test Acc 50.45%, Prec UP 50.52%, Prec DN 50.00%, p-value 0.5155, Net Return -4.22%
  - Logistic Regression: Test Acc 51.05%, Prec UP 51.32%, Prec DN 50.70%, p-value 0.3932, Net Return -1.15%
  - Random Forest: Test Acc 52.81%, Prec UP 52.10%, Prec DN 53.60%, p-value 0.1240, Net Return +1.84%
  - Gradient Boosting (Ours): Test Acc 55.01%, Prec UP 53.31%, Prec DN 57.89%, p-value 0.0362, Net Return +6.01%

- [Image: Figure 2 Results Plot - assets/results_plots.png]
  - (a) Cost-Aware Out-of-Sample Backtest (+6.01% vs -6.10% TSLA Buy & Hold, 2 bps friction)
  - (b) Real-Time Live Forward-Test on NYSE Feeds (66.7% Win Rate on 15 Resolved Trades)

**Conclusion**
The proposed Gradient Boosting framework demonstrates an authentic, statistically validated intraday edge (55.01% test accuracy, p = 0.0362, Sharpe 4.72). Live NYSE forward-testing confirms operational robustness with a 66.7% win rate, establishing that strict temporal purging and dead-zone filtering yield viable real-world algorithmic trading signals.

---

#### 5. Footer Bar

**References:**
[1] Lopez de Prado, M. (2018). Advances in Financial Machine Learning. John Wiley & Sons.
[2] Aronson, D. R. (2007). Evidence-Based Technical Analysis: Applying the Scientific Method and Statistical Inference to Trading Signals. John Wiley & Sons.
