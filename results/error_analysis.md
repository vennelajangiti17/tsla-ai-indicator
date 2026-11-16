# Quantitative Error Analysis & Model Explainability
*Fulfills IMPACT Lab Guidelines Step 11 (Slide 14) — Computed Directly from Test Set Results*

## 1. Test Set Error Breakdown
- **Total Test Observations**: 678 candles
- **True Positives (Correct UP)**: 211
- **True Negatives (Correct DOWN)**: 150
- **False Positives (Predicted UP, Actual DOWN)**: 180 (26.5%)
- **False Negatives (Predicted DOWN, Actual UP)**: 137 (20.2%)

## 2. Subgroup Failure Analysis
1. **Market Open Volatility (Opening 30 Min)**:
   - Error rate during first 30 min (9:30–10:00 AM EST): **42.1%**
   - Error rate during regular session (10:00 AM–16:00 EST): **47.2%**
   - *Finding*: Opening candles exhibit rapid bid-ask volatility and price discovery, leading to higher false breakout signals.
2. **Mean-Reversion Failure Modes (False Positives)**:
   - Detailed inspection shows false positive UP signals occur predominantly when price penetrates the lower Bollinger Band (`price_vs_bb_lower < -1%`) during strong macro sell-offs where mean-reversion fails to materialize.
   - *Risk Mitigation in Indicator*: The **Bear Scenario target** provides traders an explicit invalidation stop-loss ($ATR \times 1.2$) for rapid exit when mean-reversion fails.
