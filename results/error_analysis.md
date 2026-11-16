# Quantitative Error Analysis & Model Explainability
*Fulfills IMPACT Lab Guidelines Step 11 (Slide 14) — Computed Directly from Test Set Results*

## 1. Test Set Error Breakdown
- **Total Test Observations**: 666 regular session candles
- **True Positives (Correct UP)**: 214
- **True Negatives (Correct DOWN)**: 147
- **False Positives (Predicted UP, Actual DOWN)**: 183 (27.5%)
- **False Negatives (Predicted DOWN, Actual UP)**: 122 (18.3%)

## 2. Subgroup Failure Analysis
1. **Market Session Subgroups**:
   - Error rate during first 30 min (9:30–10:00 AM EST): **42.1%**
   - Error rate during regular session (10:00 AM–15:55 EST): **46.1%**
   - *Statistical Finding*: The opening 30 minutes shows slightly lower directional error (42.1% vs 46.1%) because strong institutional opening momentum exhibits higher directional persistence than mid-day choppy drift.
2. **Mean-Reversion Failure Modes (False Positives)**:
   - Detailed inspection shows false positive UP signals occur predominantly when price penetrates the lower Bollinger Band during strong market-wide sell-offs where technical bounces fail.
   - *Trader Risk Control*: The **Bear Scenario target** provides traders an explicit invalidation stop-loss ($ATR \times 1.2$) for swift risk mitigation.
