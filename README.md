# Intraday Price Direction Forecasting Using XGBoost on Multi-Indicator Microstructure Features for TSLA Stock

> **Coursework Project**: End-to-End Machine Learning Indicator  
> **Institution**: IMPACT Lab, Department of Computer Science & Engineering, IIT Patna  
> **Asset**: Tesla, Inc. (`NASDAQ: TSLA`) | **Timeframe**: 5-Minute Intraday Candles  
> **Horizon**: Next 2 Candles (10 Minutes) | **Target**: Directional Move ($\pm 0.08\%$)  

---

## 1. Problem Formulation (Step 1)
- **Problem Statement**: Given 35 technical, volatility, momentum, and market-microstructure features computed on 5-minute OHLCV candles, predict whether the price of **TSLA** will move **UP** ($\ge +0.08\%$) or **DOWN** ($\le -0.08\%$) over the next 10 minutes (2 candles) on live unseen market data, achieving higher precision than naive random or heuristic baselines without repainting.
- **Task Type**: Binary Time-Series Classification with Confidence Rejection (Neutral Filter).
- **Core Hypothesis**: Combining technical oscillators (RSI, MACD, Stochastics, Bollinger Bands) with market microstructure features (opening 30-min volatility, volume-to-moving-average ratio, VWAP deviation) captures short-term order imbalances in high-beta equities like TSLA.

---

## 2. Project Architecture

```
tsla_ai_indicator/
│
├── .env.example                ← Template for Alpaca API credentials
├── requirements.txt            ← Exact python environment dependencies
├── NOTES.md                    ← Ongoing experiment log & live notes
├── README.md                   ← Project documentation & submission guide
│
├── data/                       ← Historical market data & preprocessed features
│   ├── tsla_5m_historical.csv  ← 120 days of raw 5-min TSLA bars from Alpaca
│   ├── tsla_features.csv       ← 35 engineered features + binary target
│   └── tsla_day_chart.png      ← Sample trading day visualization
│
├── models/                     ← Serialized model packages
│   └── tsla_xgb_model.pkl      ← Trained XGBoost model + feature set + thresholds
│
├── results/                    ← Academic benchmarks & audit artifacts
│   ├── dataset_audit.md        ← Step 3 dataset card & data dictionary
│   ├── model_comparison.csv    ← Benchmark table comparing all baselines
│   ├── model_comparison.md     ← Formatted markdown benchmark table
│   ├── error_analysis.md       ← Step 11 failure analysis & explainability
│   ├── confusion_matrices.png  ← Confusion matrices (Logistic Reg vs XGBoost)
│   ├── feature_importance.png  ← Top 15 feature importances bar chart
│   └── eda_target_distribution.png ← Class balance visualization
│
└── src/                        ← Modular reproduction pipeline
    ├── step1_download_data.py  ← Phase 0: Download 120 days of 5-min TSLA bars
    ├── step2_explore_data.py   ← Phase 0: Exploratory Data Analysis (EDA)
    ├── step3_live_stream_test.py← Phase 0: Test live Alpaca API connectivity
    ├── step4_features_and_target.py ← Phase 1: Compute 35 features & target
    ├── step5_train_model.py    ← Phase 2: Walk-forward validation & XGBoost training
    ├── step6_scenarios.py      ← Phase 3: Multi-scenario target generator (Base/Bull/Bear)
    ├── step7_live_server.py    ← Phase 4-5: FastAPI server + WebSocket polling
    ├── step8_benchmarks_and_audit.py ← Academic evaluation & baseline comparison
    └── dashboard.html          ← Live web dashboard (TradingView Lightweight Charts)
```

---

## 3. Benchmark Results & Baseline Comparison (Steps 7, 9 & 10)

All models evaluated under an identical **chronological split** (Train: 70%, Validation: 15%, Held-Out Test: 15%) with zero data leakage:

| Model | Model Type | Val Acc (%) | Test Acc (%) | Test Prec (UP) | Test Recall | Test F1 | Train Time | Model Size |
|---|---|---|---|---|---|---|---|---|
| **Majority Class Baseline** | Trivial (Always UP) | 51.0% | 51.1% | 51.1% | 100.0% | 67.6% | 0.000s | 0.1 KB |
| **RSI Heuristic Rule** | Domain Baseline (RSI < 35 / > 65) | 49.6% | 51.6% | 51.6% | 87.4% | 64.9% | 0.000s | 0.2 KB |
| **Logistic Regression** | Conventional ML (Standardized) | 48.8% | 50.4% | 51.7% | 44.7% | 47.9% | 0.016s | 2.5 KB |
| **Proposed XGBoost (Raw)** | Gradient Boosted Trees | 50.0% | 54.9% | **62.5%** | 29.4% | 40.0% | 0.391s | 543 KB |
| **Proposed XGBoost + Confidence Filter** | High-Confidence ($P \ge 55\%$) | 50.4% | 54.5% | **62.5%** | 22.7% | 33.3% | 0.391s | 543 KB |

### Key Evaluation Insights:
- **Baseline Outperformance**: On the held-out test set, the confident XGBoost model achieves **62.5% Precision on UP signals** (+11.4 percentage points over the majority-class baseline of 51.1%).
- **Trader Economics**: In high-frequency intraday trading, predicting every move is counterproductive. By applying confidence thresholds ($P \ge 55\%$ for BUY, $P \le 45\%$ for SELL, and HOLD otherwise), the indicator filters out choppy, low-probability regimes and only alerts the trader during high-conviction momentum windows.

---

## 4. Anti-Repainting Guarantee & Live Architecture

1. **Closed-Candle Execution**:
   - The indicator polls the Alpaca API every 60 seconds.
   - It filters out the active (in-progress) candle and evaluates **strictly on completed, closed 5-minute bars**.
2. **Immutable Log Table**:
   - Once a signal is generated and assigned an ID and timestamp, it is permanently locked into the prediction log.
   - Predictions are never revised, shifted, or backfilled.
3. **Multi-Scenario Targets**:
   - **Base Target**: Most probable 10-minute price level based on $ATR \times 0.8$.
   - **Bull Target**: Optimistic upside target based on $ATR \times 1.5$ (extended up to $1.3\times$ if volume spikes $>2\times$ average).
   - **Bear Target**: Invalidation level / stop-loss based on $ATR \times 1.2$.

---

## 5. Quickstart & Reproduction Guide

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

# 3. Compute 35 technical features and chronological target
python src/step4_features_and_target.py

# 4. Train XGBoost model with walk-forward validation
python src/step5_train_model.py

# 5. Verify multi-scenario engine
python src/step6_scenarios.py

# 6. Run baseline comparisons, generate audit & benchmark tables
python src/step8_benchmarks_and_audit.py

# 7. Start live FastAPI server & dashboard
python src/step7_live_server.py
```
Open **`http://localhost:8000`** in any web browser to view the live dashboard.

---

## 6. Trader User Guide: How to Read the Indicator

### 1. Reading the Signal Badges
- **🟢 BUY (Confidence $\ge 55\%$)**: Short-term bullish momentum detected. Look for confirmation with RSI climbing above 45 and positive MACD histogram.
- **🔴 SELL (Confidence $\le 45\%$)**: Short-term bearish momentum detected. Price expected to test lower support.
- **🟡 HOLD / NEUTRAL**: Ambiguous momentum. Do not enter new directional positions; wait for conviction.

### 2. Interpreting the Scenarios
- **Base Case**: The primary target for taking profit within 10 minutes.
- **Bull / Bear Scenarios**: Dynamic volatility bands. If volume is $>2\times$ average, expect wider swings.
- **Bear (on BUY) / Bull (on SELL)**: Acts as the **hard stop-loss / invalidation line**. If price breaches this level, exit immediately.

### 3. When NOT to Trust the Indicator
1. **Major Macro News Releases**: During FOMC interest rate announcements, CPI data drops, or unexpected TSLA corporate headlines, technical patterns break down.
2. **First 5 Minutes of Market Open (9:30–9:35 AM EST)**: Opening bid-ask spreads can be wide on IEX feed, creating transient spikes.
3. **Extremely Low Volume Periods (Vol Ratio $< 0.4\times$)**: Prices drift sideways without institutional participation.
