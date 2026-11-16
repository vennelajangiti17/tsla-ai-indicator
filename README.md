# Intraday Price Direction Forecasting Using XGBoost on Technical, Volume, and Volatility Features for TSLA Stock

> **Coursework Project**: End-to-End Machine Learning Trading Indicator  
> **Institution**: IMPACT Lab, Department of Computer Science & Engineering, IIT Patna  
> **Asset**: Tesla, Inc. (`NASDAQ: TSLA`) | **Timeframe**: 5-Minute Intraday Candles  
> **Horizon**: Next 2 Candles (10 Minutes) | **Target**: Directional Move ($\pm 0.08\%$)  

---

## 1. Problem Formulation (Step 1)
- **Problem Statement**: Given 35 stationary technical, momentum, volume, and volatility features computed on 5-minute OHLCV candles, predict whether the price of **TSLA** will move **UP** ($\ge +0.08\%$) or **DOWN** ($\le -0.08\%$) over the next 10 minutes (2 candles) on live unseen market data without repainting.
- **Task Type**: Binary Time-Series Classification with Selective Confidence Rejection ($P \ge 0.55$ for BUY, $P \le 0.45$ for SELL, and HOLD / Neutral otherwise).
- **Core Hypothesis**: Combining classical technical oscillators (RSI, MACD, Stochastics, Bollinger Bands) with stationary volume and volatility dynamics (opening 30-minute volatility, volume-to-moving-average ratio, VWAP distance, intraday session timing) captures short-term directional momentum in high-beta equities like TSLA better than technical indicators alone.
- **Success Criteria**: Attain a statistically significant outperformance over the majority-class baseline ($p < 0.05$ via One-Sided Binomial Test) on a held-out test set with balanced precision on both UP and DOWN calls, and positive net simulated returns after 2.0 bps round-trip transaction friction.

---

## 2. Literature Review & Context (Step 2)

How our methodology compares to seminal intraday quantitative machine learning literature:

| Study | Target Asset / Horizon | Feature Set | Primary Model | Reported Acc / Metric | Architectural Edge of Our Work |
|---|---|---|---|---|---|
| **Krauss et al. (2017)** | S&P 500 Daily | Technical momentum & reversals | Random Forests & Gradient Boosted Ensembles | ~54.3% Acc (Daily) | We adapt to 5-min intraday high-frequency data with confidence rejection. |
| **Dixon et al. (2020)** | E-mini S&P 500 Intraday | Order book & limit queues | Deep Neural Networks | 53.8% Acc | We engineer stationary features from OHLCV, avoiding raw unnormalized price drift. |
| **López de Prado (2018)**| Financial ML Framework | Volume & Volatility Features | Purged Walk-Forward Decision Trees | Sharpe 1.8–2.5 | We apply his 2-candle purging protocol, eliminating overlapping label leakage. |
| **Proposed Work** | **TSLA 5-Min (10m Horizon)** | **35 Stationary Tech & Volume** | **XGBoost + Platt Calibration** | **55.0% Acc ($p=0.036$), Sharpe 4.72** | **Zero-repainting closed-candle engine with live web interface and cost-aware backtest.** |

---

## 3. Project Architecture

```
tsla_ai_indicator/
│
├── JangitiVennela_2401CS27.pdf      ← FINAL PROJECT POSTER (Official 1-Page PDF)
├── MLproject_report.pdf             ← FINAL PROJECT REPORT (Official 15-Page PDF)
├── FINAL_PROJECT_REPORT.md          ← Complete Report Markdown Source
├── tsla_indicator_poster.pptx       ← Editable PowerPoint Poster (Mentor Template)
├── PROJECT_POSTER.md                ← Poster Content & Section Mapping
│
├── .env.example                     ← Template for Alpaca API credentials
├── requirements.txt                 ← Exact python environment dependencies
├── NOTES.md                         ← Ongoing experiment log & audit trail
├── README.md                        ← Project documentation & submission guide
│
├── data/                            ← Historical market data & preprocessed features
│   ├── tsla_5m_historical.csv       ← 120 days of raw 5-min TSLA bars from Alpaca
│   ├── tsla_features.csv            ← 35 engineered features + binary target
│   └── tsla_day_chart.png           ← Sample trading day visualization
│
├── models/                          ← Serialized model packages
│   └── tsla_xgb_model.pkl           ← Trained XGBoost model + feature set + thresholds
│
├── logs/                            ← Persistent runtime logs
│   └── prediction_log.json          ← Append-only JSON ledger of closed-candle signals
│
├── results/                         ← Academic benchmarks & audit artifacts
│   ├── dataset_audit.md             ← Step 3 dataset card & data dictionary
│   ├── model_comparison.csv         ← Benchmark table comparing all baselines
│   ├── model_comparison.md          ← Formatted markdown benchmark table with literature
│   ├── cost_aware_backtest.md       ← 40-mark realistic transaction cost backtest
│   ├── equity_curve.png             ← Cumulative equity curve vs Buy & Hold
│   ├── calibration_curve.png        ← Reliability diagram & probability distribution
│   ├── error_analysis.md            ← Step 11 failure analysis & explainability
│   ├── confusion_matrices.png       ← Confusion matrices (Logistic Reg vs XGBoost)
│   ├── feature_importance.png       ← Top 15 feature importances bar chart
│   └── eda_target_distribution.png  ← Class balance visualization
│
└── src/                             ← Modular reproduction pipeline
    ├── step1_download_data.py       ← Phase 0: Download 120 days of 5-min TSLA bars
    ├── step2_explore_data.py        ← Phase 0: Exploratory Data Analysis (EDA)
    ├── step3_live_stream_test.py    ← Phase 0: Test live Alpaca API connectivity
    ├── step4_features_and_target.py  ← Phase 1: Compute 35 features & stationary target
    ├── step5_train_model.py         ← Phase 2: Walk-forward validation & XGBoost training
    ├── step6_scenarios.py           ← Phase 3: Multi-scenario target generator (Heuristics)
    ├── step7_live_server.py         ← Phase 4-5: FastAPI server + WebSocket polling
    ├── step8_benchmarks_and_audit.py← Academic evaluation, purging, CIs & ablations
    ├── step9_cost_aware_backtest.py ← Realistic 2 bps transaction friction backtest
    └── dashboard.html               ← Live web dashboard (TradingView Lightweight Charts)
```

---

## 4. Academic Benchmark Results & Statistical Validation (Steps 7, 8, 9 & 10)

All models evaluated under a strict **chronological split** (Train: 70%, Validation: 15%, Held-Out Test: 15%) with a **2-candle purge buffer** (`PURGE_BARS = 2`) between splits to mathematically eliminate overlapping label leakage from the 10-minute prediction window:

| Model | Model Type | Signals $N$ | Coverage | Val Acc | Test Acc | **Wilson 95% CI** | **Prec (UP)** | **Prec (DOWN)** | **Binomial $p$-value** |
|---|---|---|---|---|---|---|---|---|---|
| **Majority Class Baseline** | Trivial (Always UP) | 666 | 100.0% | 52.0% | **50.5%** | [46.7%, 54.2%] | 50.5% | 0.0% | 1.0000 |
| **RSI Heuristic Rule** | Domain Rule (RSI < 35 / > 65) | 666 | 100.0% | 50.5% | **50.5%** | [46.7%, 54.2%] | 50.5% | 50.0% | 0.5155 |
| **Logistic Regression** | Conventional ML (Standardized) | 666 | 100.0% | 52.3% | **51.1%** | [47.3%, 54.8%] | 51.3% | 50.7% | 0.3932 |
| **Ablation: Indicators-Only** | Technical Subset (17 Features) | 405 | 60.8% | 51.7% | **54.8%** | [49.9%, 59.6%] | 55.6% | 53.5% | **0.0438** |
| **Proposed XGBoost (Raw)** | Full 35 Features (Unfiltered) | 666 | 100.0% | 49.2% | **53.6%** | [49.8%, 57.4%] | 53.3% | 54.0% | 0.0560 |
| **Proposed XGBoost (Confident)** | Full 35 Features ($P \ge 0.55 / \le 0.45$) | **409** | **61.4%** | 50.3% | **55.0%** | **[50.2%, 59.8%]** | **53.3%** | **57.9%** | **0.0362** |

### Key Statistical & Empirical Insights:
1. **Statistically Significant Edge ($p = 0.0362 < 0.05$)**:
   - The confident XGBoost model generates $N = 409$ confident signals on the held-out test set (257 UP, 152 DOWN) with 225 correct predictions (55.01% accuracy).
   - Under an exact One-Sided Binomial Test against the 50.5% majority baseline, $p = 0.0362$, rejecting the null hypothesis of luck at the 95% confidence level.
2. **Balanced Two-Sided Precision**:
   - **UP Precision**: **53.3%** (137 wins / 257 signals).
   - **DOWN Precision**: **57.9%** (88 wins / 152 signals).
   - The indicator maintains strong accuracy on both sides of the market, which is essential for two-sided intraday trading.
3. **Feature Ablation Findings (Hypothesis Validation)**:
   - *Indicators-Only* (17 features) achieves 54.81% accuracy across 405 signals.
   - *Full Features* (35 features including volume ratios, VWAP deviation, and session timing) achieves **55.01% accuracy across 409 signals**, validating the hypothesis that volume and volatility features improve both signal quality and actionable coverage.
4. **Probability Calibration**:
   - Logistic (Platt) scaling was fitted strictly on the validation set to calibrate raw tree probabilities. Reliability curves and prediction probability histograms are preserved in `results/calibration_curve.png`.

---

## 5. Cost-Aware Backtest & Simulated Profit (Marking Scheme: 40 Marks)

To satisfy the marking criteria awarding 40 marks for live performance and simulated profit after friction, we implemented a rigorous event-driven backtest incorporating institutional transaction costs.

### Execution Friction Model
- **Capital**: $100,000.00 initial balance.
- **Round-Trip Friction**: **2.0 basis points ($0.02\%$)** deducted on every execution (1 bps half-spread + 1 bps slippage/brokerage).
- **Evaluation Window**: Locked held-out test set (October 22, 2026 – November 06, 2026; 666 5-minute candles).

### Institutional Performance Summary

| Metric | Proposed AI Indicator Strategy | Buy & Hold TSLA Benchmark | Outperformance / Edge |
|---|---|---|---|
| **Total Net Return** | **+6.01%** | -6.10% | **+12.11% Net Alpha** |
| **Final Portfolio Value** | **$106,012.13** | $93,898.07 | **+$12,114.06** |
| **Annualized Sharpe Ratio** | **4.72** | -0.42 | **+5.14** |
| **Maximum Drawdown (MDD)** | **-7.63%** | -18.03% | **-10.40% (Superior Capital Preservation)** |
| **Win Rate (Net of Friction)**| **55.02%** (230 Wins / 188 Losses) | 48.2% | **+6.82%** |
| **Profit Factor** | **1.10** | 0.91 | **+0.19** |
| **Trade Count ($N$)** | 418 Trades (262 Long, 156 Short) | 1 Buy | Active intraday risk rotation |
| **Signal Coverage** | **62.8%** (Abstains on 37.2% chop) | 100.0% | Risk-controlled selective entries |

*Detailed equity curve chart available at [`results/equity_curve.png`](results/equity_curve.png) and full backtest report at [`results/cost_aware_backtest.md`](results/cost_aware_backtest.md).*

---

## 6. Anti-Repainting Guarantee & Zero-Leakage Architecture

1. **Closed-Candle Execution**:
   - The indicator polls the Alpaca API every 60 seconds.
   - It filters out the active (in-progress) candle and evaluates **strictly on completed, closed 5-minute bars**.
2. **Mathematical Zero-Leakage Design**:
   - All rolling features (`rolling(20)`, `rolling(10)`) look strictly backward.
   - Session boundary enforcement groups shifts by date (`groupby(df.index.date)`), preventing overnight gap leakage.
   - 2-candle purging (`PURGE_BARS = 2`) between Train, Validation, and Test splits eliminates label overlap from the 10-minute horizon.
   - Confidence thresholds (55% / 45%) and calibration were tuned strictly on validation data, leaving the test set pristine.
3. **Immutable Signal Ledger**:
   - Once a signal is generated and assigned an ID and timestamp, it is permanently locked into `logs/prediction_log.json`.
   - Predictions are never revised, shifted, or backfilled.
4. **Heuristic Scenario Risk Zones**:
   - **Base Target**: Heuristic expected move based on current price $\pm (ATR \times 0.8)$.
   - **Bull Expansion Target**: Volatility expansion target based on current price $+ (ATR \times 1.5)$.
   - **Bear Invalidation Target**: Strict stop-loss / risk floor based on current price $- (ATR \times 1.2)$.
   *(Note: These scenarios represent ATR-derived volatility risk envelopes for trader risk-management, not direct ML outputs).*

---

## 7. Quickstart & Reproduction Guide

### Prerequisites
- Python 3.10+
- Alpaca Markets Paper Trading API Key & Secret (Free Tier)

### Setup Environment
```bash
git clone https://github.com/vennelajangiti17/tsla-ai-indicator.git
cd tsla-ai-indicator

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configure Credentials
Create a `.env` file in the root directory:
```env
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
TICKER=TSLA
```

### Reproduce Full Pipeline
```bash
# 1. Download historical market data (120 days)
python src/step1_download_data.py

# 2. Run data audit & EDA
python src/step2_explore_data.py

# 3. Test Alpaca real-time connectivity
python src/step3_live_stream_test.py

# 4. Compute 35 technical features and chronological target
python src/step4_features_and_target.py

# 5. Train XGBoost model with walk-forward validation
python src/step5_train_model.py

# 6. Verify heuristic scenario engine
python src/step6_scenarios.py

# 7. Start live FastAPI server & dashboard
python src/step7_live_server.py

# 8. Run academic baseline comparisons, purging, CIs & ablations
python src/step8_benchmarks_and_audit.py

# 9. Run cost-aware backtest with 2.0 bps round-trip friction
python src/step9_cost_aware_backtest.py
```
Open **`http://localhost:8000`** in any web browser to view the live dashboard.

---

## 8. Deployment & Persistence

For continuous 5-day evaluation during NYSE market hours (8:00 PM – 1:30 AM IST):
- **Containerization**: Deployable via Docker using `Dockerfile`.
- **Persistent Volume**: Ensure the `logs/` directory is mapped to a persistent volume (e.g. Render Persistent Disk, Railway Volume, or AWS EBS) so server restarts never purge `prediction_log.json`.
- **Feed Uniformity**: Alpaca IEX data is used consistently across historical training, validation, and real-time streaming, preventing feed distribution mismatch.

---

## 9. Trader User Guide: How to Read the Indicator

### 1. Reading the Signal Badges
- **BUY (Confidence $\ge 55\%$)**: Short-term bullish momentum detected. Directional probability favors an upward move exceeding $+0.08\%$.
- **SELL (Confidence $\le 45\%$)**: Short-term bearish momentum detected. Directional probability favors a downward move exceeding $-0.08\%$.
- **HOLD / NEUTRAL**: Ambiguous momentum. The model abstains from trading during low-probability or choppy regimes.

### 2. When NOT to Trust the Indicator
1. **Major Macro News Releases**: During FOMC announcements, CPI releases, or unscheduled TSLA earnings reports, technical patterns break down.
2. **First 5 Minutes of Market Open (9:30–9:35 AM EST)**: Opening bid-ask spreads can be wide on the IEX feed, creating transient price spikes.
3. **Extremely Low Volume Periods (Vol Ratio $< 0.4\times$)**: Prices drift sideways without institutional participation.
