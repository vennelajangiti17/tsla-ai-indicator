# Error Analysis, Explainability and Robustness
*Fulfills IMPACT Lab Guidelines Step 11 (Slide 14)*

## 1. Top Predictive Factors
As revealed by feature importance analysis:
1. `is_first_30min` (Opening Volatility): The first 30 minutes of NYSE trading (9:30 AM – 10:00 AM EST) have significantly different distribution dynamics than mid-day sessions. The model relies heavily on this time flag to scale risk.
2. `bb_mid` and `bb_upper` (Bollinger Mean Reversion): Price displacement relative to moving bands is a primary driver for TSLA mean-reversion signals.
3. `ema_cross` and `macd_hist`: Momentum confirmation prevents counter-trend fading during strong directional drives.

## 2. Failure Case Analysis (False Positives & False Negatives)
- **False Positives (Predicted UP, actual DOWN)**:
  - *Context*: Primarily occur during high volume breakout attempts that quickly get rejected (false bull breakouts) near local resistance.
  - *Mitigation in Dashboard*: The Bear Scenario explicitly calculates the invalidation/stop-loss level ($ATR \times 1.2$), warning traders of rapid reversals.
- **False Negatives (Predicted DOWN, actual UP)**:
  - *Context*: Commonly occur during sudden external market catalysts (e.g., broad NASDAQ rallies, Fed interest rate announcements, or sudden tweet events) that override technical indicators.
  - *Mitigation*: The confidence filter labels ambiguous signals as NEUTRAL (HOLD), preventing overtrading.

## 3. Robustness & Anti-Repainting Verification
- **Anti-Repainting Guarantee**: All indicators and predictions are strictly computed on **closed** 5-minute candles. Once a signal is broadcast and logged, it is permanently locked with its timestamp.
- **Live Testing Verification**: During live testing, incoming real-time candles are evaluated sequentially with identical feature pipelines, ensuring zero lookahead leakage.
