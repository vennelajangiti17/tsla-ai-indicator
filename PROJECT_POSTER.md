# AI-Powered Intraday Trading Indicator for Tesla (TSLA) Using XGBoost
## B.Tech CSE Project Poster Presentation

### Poster Metadata
- Title: AI-Powered Intraday Trading Indicator for Tesla (TSLA) Using XGBoost
- Author: Jangiti Vennela
- Roll Number: 2401CS27
- Email: jangiti_2401cs27@iitp.ac.in
- Affiliation: Department of Computer Science and Engineering, Indian Institute of Technology Patna

---

### Poster Sections & Content

#### 1. Header Bar
- Top Left: Indian Institute of Technology Patna Official Logo
- Center:
  - Line 1: AI-Powered Intraday Trading Indicator for Tesla (TSLA) Using XGBoost
  - Line 2: Jangiti Vennela, Roll No: 2401CS27, jangiti_2401cs27@iitp.ac.in
  - Line 3: Department of Computer Science and Engineering, Indian Institute of Technology Patna
- Top Right: IMPACT Lab Official Logo

---

#### 2. Left Column

**Abstract**
Fixed-formula indicators such as RSI and MACD only describe past prices. This project trains an XGBoost model on 5-minute TSLA candles to predict a move of at least 0.08% over the next 10 minutes, and shows UP, DOWN or HOLD with a confidence on a live dashboard. Confident signals reach 55.0% accuracy on the held-out test set (95% CI 50.2% to 59.8%) against 50.5% for always guessing UP. Signals use closed candles only and are logged, so they never repaint.

**Introduction**
- Intraday prices are noisy and close to a coin toss, so fixed rules give many false signals in quiet markets.
- Overlapping labels and shuffled splits leak future data into training, so a time-ordered split with purge gaps is used [2].
- The model stays out (HOLD) when unsure, and every signal is scored against what price did 10 minutes later.

**Problem Statement**
Predict whether TSLA will move up or down by at least 0.08% in the next 10 minutes from 35 stationary features of 5-minute candles, without look-ahead leakage or repainting, and give no signal when confidence is low.

---

#### 3. Center Column

**Key Contributions**
- A complete live indicator: data download, 35 stationary features, an XGBoost model [1], a FastAPI server and a Lightweight Charts dashboard [4] with a signal log.
- Leak-free design: session-aware 10-minute labels, a flat-move filter, a 70/15/15 time split with 2-bar purge gaps, and settings chosen on validation data only.
- Honest testing against three baselines with Wilson confidence intervals, a calibration check, a cost-aware paper test and a live session.

**Methodology**
- Figure 1 Architecture Diagram (`assets/architecture_diagram.png`) with 4 colored stages:
  1. Data: Alpaca IEX [3], 5-min TSLA candles, 13 Jul to 6 Nov 2026, 4,438 labelled rows, flat-move filter
  2. Features: 35 stationary features, RSI, MACD, Bollinger, ATR, VWAP, Label: 10-min move, ±0.08%
  3. Model: XGBoost [1], 300 trees, depth 3, 70/15/15 time split, 2-bar purge, Platt calibration
  4. Live System: FastAPI server, polls every 60 s, closed candles only, Lightweight Charts [4], disk log
- Caption: Figure 1: End-to-end architecture of the TSLA indicator

**Algorithm 1: TSLA 10-Minute Direction Signal**
- Require: 5-min OHLCV candles D; flat threshold δ = 0.08%; horizon k = 2; cut-offs 0.55 / 0.45
- Ensure: Signal in {UP, DOWN, HOLD} with a confidence
- 1. Compute 35 stationary features at each candle close.
- 2. Label UP if the 2-candle return is ≥ +δ, DOWN if ≤ −δ; drop flat rows.
- 3. Split by time with purge gaps; train XGBoost; tune and calibrate on validation.
- 4. On each closed candle: p = P(UP). UP if p ≥ 0.55, DOWN if p ≤ 0.45, else HOLD.
- 5. Log the signal; after 10 minutes, score it against the ±δ rule.
- 6. Return the model, test metrics and the signal log.

---

#### 4. Right Column

**Result and Discussion**
- Table 1 Caption: Table 1: Held-out test results (666 candles)
- Table 1 Data:
  - Always UP: Signals 666, Test acc. 50.5%, 95% CI 46.7–54.2%, p-value 1.000
  - RSI rule: Signals 666, Test acc. 50.5%, 95% CI 46.7–54.2%, p-value 0.516
  - Logistic regression: Signals 666, Test acc. 51.1%, 95% CI 47.3–54.8%, p-value 0.393
  - XGBoost (raw, 0.50): Signals 666, Test acc. 53.6%, 95% CI 49.8–57.4%, p-value 0.056
  - XGBoost (confident): Signals 409, Test acc. 55.0%, 95% CI 50.2–59.8%, p-value 0.036*
  - Footnote: p-value: one-sided binomial test against the 50.5% always-UP rate. Confident signals: UP precision 53.3%, DOWN precision 57.9%.
- Figure 2: Paper test over the 15-day test window (2 bps round-trip cost)
  - Strategy net return +6.01% vs -6.10% TSLA buy-and-hold; Max drawdown -7.6% vs -18.0%
- Figure 3: Live dashboard, 16 Nov 2026
  - 15 scored signals: 10 wins, 5 losses (66.7% win rate); logged signals never changed (zero repainting)

**Conclusion**
The model gives a small but measurable edge: 55.0% on 409 confident test signals (95% CI 50.2–59.8%, p = 0.036). The paper test and the live session were positive but are small samples from one market regime, and probabilities above 55% were over-confident (about 60% predicted, 53% actual).

---

#### 5. Footer Bar

**References:**
- [1] Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proc. 22nd ACM SIGKDD, pp. 785–794.
- [2] López de Prado, M. (2018). Advances in Financial Machine Learning. John Wiley & Sons.
- [3] Alpaca Markets (2026). Alpaca Market Data API v2 documentation. alpaca.markets/docs
- [4] TradingView (2026). Lightweight Charts documentation. tradingview.github.io/lightweight-charts
