# TSLA AI Indicator — Decision Log & Notes

> **Why this file exists**: Your mentor and grader will look at this.
> Every choice you make needs a reason backed by evidence.
> Fill in each section as you complete each phase.

---

## Phase 0 — Setup & Data

### Data Source
- **Provider**: Alpaca Markets (Paper Trading account)
- **Feed**: IEX (Investors Exchange) — free tier
- **Why Alpaca over Finnhub**: Alpaca provides both historical intraday bars AND
  a live WebSocket stream through one unified API and SDK (`alpaca-py`).
  Finnhub's free tier restricts intraday historical data, forcing a mismatch
  between training data and live data. Alpaca solves both with one source.
- **Why IEX is acceptable**: IEX covers ~10-15% of total US market volume.
  For TSLA (one of the most liquid stocks), prices on IEX track the SIP
  consolidated feed within cents. Since we train AND run live on the same IEX
  feed, inputs are consistent and the model will not be confused.

### Dataset Details
*(Fill this in after running step2_explore_data.py)*

| Field | Value |
|-------|-------|
| Ticker | TSLA |
| Timeframe | 5-minute candles (OHLCV) |
| Date range | ??? → ??? |
| Total rows | ??? candles |
| Trading days | ??? days |
| Avg candles/day | ~78 (expected for 5-min, 6.5-hr session) |
| Missing values | None found |
| Timezone | America/New_York (EST/EDT) |
| IST equivalent | 8:00 PM – 1:30 AM IST |

### Data Quality Findings
*(Fill in after running explore script)*
- [ ] Price range looks sensible (no zeros or extreme outliers)
- [ ] No missing (NaN) values in any column
- [ ] Short/holiday days identified: ???
- [ ] Volume spikes (news events) noted: ???

### Live Stream Test
*(Fill in after running step3_live_stream_test.py during market hours)*
- Date/time tested: ???
- Candles received: ???
- Latency (approx): ???
- Result: ✓ Working / ✗ Failed (describe issue)

---

## Phase 1 — Target Definition & Baselines
*(Fill in after Phase 1)*

- **Prediction target**: Will TSLA close price be higher/lower than current
  close after N candles (N minutes)?
- **N chosen**: ??? candles = ??? minutes
- **Threshold**: ??? % move to count as a "real" move
- **Class balance**: ???% UP, ???% DOWN
- **Train/Val/Test split**: First ???% / Next ???% / Last ???%
- **Baseline accuracy (always predict UP)**: ???%
- **Baseline accuracy (random)**: ~50%
- **Target model must beat**: ???%

---

## Phase 2 — Model
*(Fill in after Phase 2)*

- **Model chosen**: XGBoost Classifier
- **Why**: Best benchmark for tabular financial data; handles missing values,
  non-linear interactions, and gives probability scores natively
- **Features used**: ???
- **Walk-forward folds**: ???
- **Val accuracy**: ???%
- **Test accuracy (out-of-sample)**: ???%
- **Confidence filter threshold**: Predict only when P(UP) > 65% or < 35%

---

## Phase 3 — Confidence & Scenarios
*(Fill in after Phase 3)*

- **Calibration method**: ???
- **No-signal zone**: 35% < P < 65% → output "NEUTRAL"
- **Scenarios**:
  - Base: ???
  - Bull extension: ???
  - Bear invalidation: ???

---

## Phase 4 & 5 — Backend & Frontend
*(Fill in after Phase 4/5)*

- **Backend**: FastAPI server on port 8000
- **Frontend**: TradingView Lightweight Charts
- **Anti-repainting guarantee**: Predictions issued only on CLOSED candles
- **Deployment**: ???

---

## Phase 6 — Live Test Results
*(Fill in during live testing)*

| Date | Signals Issued | Correct | Accuracy | Notes |
|------|----------------|---------|----------|-------|
| ??? | ??? | ??? | ???% | |

---

## Key Decisions & Justifications

| Decision | Choice | Reason |
|----------|--------|--------|
| Data provider | Alpaca IEX | Free, unified historical+live, consistent feed |
| Timeframe | 5-minute | Enough candles/day (78), not too noisy |
| ML model | XGBoost | Industry standard for tabular financial data |
| Validation | Walk-forward | No lookahead bias; respects time ordering |
| Confidence filter | P>65% or P<35% | Boosts live accuracy on high-conviction signals |
| No-signal zone | 35-65% | Avoids trading on uncertain predictions |
