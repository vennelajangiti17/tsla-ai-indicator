# Quantitative Error Analysis & Model Explainability
*Fulfills IMPACT Lab Guidelines Step 11 (Slide 14) — Computed Directly from Test Set Results*

## 1. Test Set Error Breakdown
- **Total Test Observations**: 666 regular session candles
- **True Positives (Correct UP)**: 215
- **True Negatives (Correct DOWN)**: 142
- **False Positives (Predicted UP, Actual DOWN)**: 188 (28.2%)
- **False Negatives (Predicted DOWN, Actual UP)**: 121 (18.2%)

## 2. Subgroup Failure Analysis
1. **Market Session Subgroups**:
   - Error rate during first 30 min (9:30–10:00 AM EST): **38.6%**
   - Error rate during regular session (10:00 AM–15:55 EST): **47.1%**
   - *Statistical Finding*: The opening 30 minutes shows slightly lower directional error (38.6% vs 47.1%) because strong institutional opening momentum exhibits higher directional persistence than mid-day choppy drift.
2. **Mean-Reversion Failure Modes (False Positives)**:
   - Detailed inspection shows false positive UP signals occur predominantly when price penetrates the lower Bollinger Band during strong market-wide sell-offs where technical bounces fail.
   - *Trader Risk Control*: The **Bear Scenario target** provides traders an explicit invalidation stop-loss ($ATR \times 1.2$) for swift risk mitigation.
