Intraday Price Direction Prediction for TSLA Using XGBoost on 5-Minute Volume and Volatility Features

Name: Jangiti Vennela
Roll No: 2401CS27
Course: End-to-End Machine Learning Project, IMPACT Lab, Dept. of CSE, IIT Patna
Repository: https://github.com/vennelajangiti17/tsla-ai-indicator


Abstract

Traders commonly use indicators like RSI and MACD, but these tools only summarize past price movements using fixed formulas. In this project, I built an intraday indicator powered by a machine learning model. The model looks at 5-minute candles of Tesla (TSLA) and predicts whether the price is likely to go UP by at least 0.08%, or go DOWN by at least 0.08%, over the next 10 minutes (2 candles ahead). When the model is not confident, it emits a HOLD signal, which means low confidence and tells the trader to stay out of the market.

I trained an XGBoost model on 120 calendar days of 5-minute data from the Alpaca Markets API. The feature pipeline computes 35 stationary features from price, volume, and volatility. To avoid looking into the future, the data was split in time order, and a 2-candle purging gap was added between splits so overlapping 10-minute labels do not leak. I compared the model with three simple baselines: always predicting UP, a basic RSI rule, and logistic regression. The indicator runs live on a custom web app and only evaluates closed candles, so a signal is locked and never repaints.

On the held-out test set of 666 candles, the model produced 409 confident signals with an accuracy of 55.01% (95% Wilson confidence range of 50.2% to 59.8%). Precision on UP signals was 53.3% and precision on DOWN signals was 57.9%. An exact one-sided binomial test against the 50.5% majority baseline gave a p-value of 0.0362, showing that this edge is statistically distinguishable from luck at the 95% level. In a cost-aware backtest with 2.0 basis points of round-trip trading friction, the strategy made +6.01% net return while Tesla stock fell -6.10%.

My honest summary: Short-term price direction is very difficult to predict and close to a coin toss. The results show a small, realistic edge of about 4.5 percentage points over the baseline. The main value of the indicator comes from abstaining during low-confidence choppy periods, which avoids whipsaw losses.


Contributions

1. Complete working pipeline: automated scripts for downloading data, building features, model training, validation, backtesting, and a real-time web dashboard.
2. Fair baseline comparison: compared against an always-UP rule, an RSI threshold rule, and logistic regression on the exact same time split and metrics.
3. Guaranteed zero repainting: signals evaluate only on completed 5-minute candles and are saved immediately to a permanent log file on disk.
4. Honest validation checks: implemented a 2-candle purge buffer, Platt probability calibration, confidence intervals, a binomial significance test, and an ablation study.


1. Introduction and Problem Statement

1.1 Why this problem
Indicators like RSI, MACD, and Bollinger Bands use fixed formulas written by humans. They do not learn from market data and cannot tell a trader the probability of what will happen next. This project asks us to replace fixed rules with a machine learning model that learns from past market data and provides a calibrated probability.

1.2 Problem statement
Given 35 features computed from 5-minute TSLA candles (price, volume, volatility, and momentum), predict whether the price will move UP by at least 0.08%, or DOWN by at least 0.08%, over the next 2 candles (10 minutes). The model emits a directional signal only when it is confident (probability >= 55% for UP, <= 45% for DOWN), and emits HOLD otherwise, indicating low confidence.
- Task type: binary classification on financial time series, with a selective filter zone for low confidence.
- Asset: TSLA (NASDAQ), single equity only.
- Timeframe and horizon: 5-minute candles, predicting 2 candles ahead (10 minutes).
- Why 0.08%?: On a 380 dollar stock like Tesla, 0.08% is about 30 cents. Moves smaller than 30 cents over 10 minutes are mostly spread and bid-ask noise. Because round-trip transaction costs are about 2 basis points (0.02%), targeting moves above 0.08% ensures the trade is larger than the friction.

1.3 Success criteria (set during development)
- On the untouched test set, directional accuracy on confident signals should be at least 54.0% across at least 200 signals.
- It should beat the always-UP baseline with statistical significance (p < 0.05 via an exact one-sided binomial test).
- The confidence shown on the dashboard should match actual win rates (verified by a calibration check).
- The strategy must produce a positive net return after deducting 2.0 basis points of round-trip trading friction.
- The live indicator must never repaint and must run stably during market hours.

1.4 Main idea (hypothesis)
Standard technical indicators often give false breakout signals during low-volume sideways trading. I hypothesize that combining technical indicators with volume and volatility features (volume ratio compared to its 20-period average, distance from session VWAP, and session timing) helps the model filter out false moves and detect true short-term momentum in a volatile stock like TSLA. I test this idea directly with an ablation study in Section 7.


2. Related Work and Gap

Table 1: Related Work Summary

| Study | Target Data | Method | How It Was Tested | Limitation |
|---|---|---|---|---|
| Fischer and Krauss (2018) | Daily S&P 500 stocks (1992 to 2015) | LSTM Deep Neural Networks | Walk-forward in time | Daily data only; trading costs can wipe out the daily edge |
| Chen and Guestrin (2016) | General tabular benchmarks | XGBoost | Standard cross-validation | General algorithm; does not handle financial label overlap out of the box |
| Lopez de Prado (2018) | Financial ML framework | Purged walk-forward time splits | Methodological guide | Requires custom implementation of purging and stationarity |
| Dixon et al. (2020) | High-frequency E-mini S&P futures | Deep Neural Networks on order book queues | Chronological test split | Requires expensive Level 2 feeds not easily accessible for retail setups |

Gap: Many student projects show a flattering backtest using random train-test splits (which leak future data), report no confidence ranges, and never run a live forward test. This project focuses on realistic methodology: a strict time split with a 2-candle purge, honest confidence intervals, simple baselines, a cost-aware backtest, and a live non-repainting web indicator.


3. Dataset and Quality Check

3.1 Source
- Source: Alpaca Markets API (paper-trading account), IEX data feed.
- Asset and period: TSLA, 5-minute candles, July 13, 2026 to November 06, 2026 (about 120 calendar days, 83 trading sessions).
- Raw data size: 6,904 candles.
- Clean feature dataset: 4,438 candles. The drop from 6,904 to 4,438 is mostly due to removing flat chop rows where the 10-minute return was within plus or minus 0.08% (about 2,250 rows), plus 166 candles dropped at the end of trading days so shifts do not cross into the next day, and 50 warmup candles for 20-period indicators.
- Why this source: the live web app reads candles from Alpaca, so training and live inference use the exact same feed.

3.2 Important note about volume
The IEX feed covers a small part of total national equity volume (around 2% to 4%). For Tesla, prices track the consolidated market within pennies, but volume numbers are smaller and noisier. This is acceptable because training and live streaming use the same feed, and I use volume ratios rather than raw share counts.

3.3 Checks performed

Table 2: Data Quality Checks

| Quality Check | What I Found | Resolution / Verification |
|---|---|---|
| Missing candles | 0 missing candles | Verified regular NYSE hours (9:30 AM to 4:00 PM ET) |
| Duplicate rows | 0 duplicate timestamps | Verified unique timestamp index |
| Impossible values | Zero negative prices, volumes, or high < low | Passed all sanity checks |
| Extreme spikes | High volatility around 9:30 AM open | Normalized using rolling 14-period ATR |
| Target balance | 2,244 UP (50.6%) vs 2,194 DOWN (49.4%) | Naturally balanced; no synthetic balancing required |

[INSERT SCREENSHOT: Target Class Distribution Plot (results/eda_target_distribution.png)]
Caption: Figure 1 shows that the clean binary target classes are naturally balanced (50.6% UP vs 49.4% DOWN), so plain accuracy is a fair metric.


4. Split Plan and Methodology

4.1 Time-based split
Data was split in time order. I never shuffled, because shuffling lets the model look into the future.

Table 3: Chronological Data Splits

| Split Block | Share | Period | Candle Count | Purpose |
|---|---|---|---|---|
| Training | 70% | July 13 to October 02, 2026 | 3,104 bars | Fit XGBoost trees |
| Purge Gap 1 | 2 bars | Boundary | 2 bars | Prevents label overlap into validation |
| Validation | 15% | October 02 to October 22, 2026 | 664 bars | Hyperparameter tuning and Platt calibration |
| Purge Gap 2 | 2 bars | Boundary | 2 bars | Prevents label overlap into test |
| Test | 15% | October 22 to November 06, 2026 | 666 bars | Single evaluation pass for final report |
| Total Clean Rows | 100% | July 13 to November 06, 2026 | 4,438 bars | Non-null binary target rows |

4.2 Gap between parts
Because each label looks 2 candles (10 minutes) into the future, the last two candles of training have targets calculated from prices inside validation. To stop this leak, I left a gap of 2 candles between training, validation, and test.

4.3 Leakage checks
- Every feature at candle t uses only data up to the close of candle t.
- Day-level features like opening 30-minute range and session VWAP are computed cumulatively from 9:30 AM of that day only.
- Rolling features look strictly backward.
- Confidence thresholds and model settings were chosen on validation data only, never on test data.
- The test set was used exactly once for the final numbers.

4.4 Features (35 in total)
- Momentum: RSI 14, MACD percentage, MACD signal percentage, MACD histogram, Stochastic K and D, Rate of Change (5 and 10), past candle returns (lags 1, 2, 3).
- Volatility: ATR percentage, Bollinger Band width, Bollinger percentage b, price distance to upper, lower, and middle bands.
- Volume: volume ratio (volume divided by 20-period average volume), On-Balance Volume z-score, price distance from VWAP.
- Moving average trends: price distance to 9 EMA and 21 EMA, EMA cross percentage, price distance to 20-candle high and low.
- Candle geometry and timing: candle body size, upper shadow, lower shadow, trading hour, opening 30-minute flag, closing 30-minute flag, power hour flag.

4.5 How the label is made
For candle t, the return is calculated as (Close[t+2] - Close[t]) / Close[t].
- If return >= +0.08%, label is UP (Class 1).
- If return <= -0.08%, label is DOWN (Class 0).
- If return is between -0.08% and +0.08%, it was removed during training because it represents flat chop. In the live test, these instances resolve as FLAT (breakeven), which means no money was gained or lost.

4.6 Algorithm
Name: TSLA 10-Minute Direction Prediction with Confidence Filter
1. Download 5-minute TSLA candles from Alpaca; compute the 35 stationary features.
2. Form the 10-minute directional target with session grouping and 2-candle purging.
3. Split the data chronologically (70% Train, 15% Val, 15% Test).
4. Train XGBoost on the training set; tune settings on validation.
5. Calibrate probabilities on the validation set using Platt scaling.
6. Set confidence thresholds (BUY >= 55%, SELL <= 45%, HOLD otherwise).
7. Run one time on the test set; record accuracy, precision, and confidence intervals.
8. Deploy live: poll Alpaca every 60s, evaluate closed bars, append to log, stream to dashboard.

4.7 Architecture diagram
Alpaca 5m Data -> 35 Features -> Purged Time Split (70/15/15) -> Train XGBoost -> Tune & Calibrate on Val -> Test Once on Test Set.
Live Engine: Alpaca Polling (60s) -> Closed Bars Only -> Features -> Saved Model -> Signal Log (Locked on Disk) -> FastAPI -> Web Dashboard.


5. Experimental Setup and Baselines

5.1 Models compared

Table 4: Baseline Models

| Model | Model Type | Why Included |
|---|---|---|
| Always UP | Trivial Heuristic | The simplest guess; tests whether the model is just riding upward drift |
| RSI Rule | Domain Rule | Standard technical rule (BUY when RSI < 35, SELL when RSI > 65, HOLD otherwise) |
| Logistic Regression | Conventional Linear ML | Simple linear model with standardized features and L2 regularization |
| XGBoost (Raw 0.50) | Gradient Boosted Trees | Full tree ensemble evaluating all test candles with default 0.50 cutoff |
| XGBoost (Confident) | Tree Ensemble + Filter | Proposed model emitting signals only when probability is >= 55% or <= 45% |

5.2 Setup details
- Same chronological split and metrics for every model.
- Random seed: 42. Python 3.10, xgboost 3.4.1, scikit-learn 1.9.1, pandas 3.1.0.
- Hardware: Apple M-series MacBook Air, 8-core CPU.
- Hyperparameter tuning: tried max_depth [2, 3, 4, 5], n_estimators [100, 200, 300, 500], learning_rate [0.01, 0.03, 0.05, 0.10], min_child_weight [5, 10, 15, 20]. Best validation settings: max_depth=3, n_estimators=300, learning_rate=0.03, min_child_weight=15, subsample=0.8, colsample_bytree=0.8, gamma=0.5, reg_alpha=0.5, reg_lambda=2.0.

5.3 Metrics
I use accuracy, precision on UP, precision on DOWN, and Wilson 95% confidence intervals. Precision matters most because a trader acts only on confident signals.


6. Results

6.1 Main comparison table (Held-Out Test Set: 666 bars)

Table 5: Model Performance Comparison

Model: Always UP
Signals N: 666
Coverage: 100.0%
Val Acc: 52.0%
Test Acc: 50.5%
Wilson 95% CI: [46.7%, 54.2%]
Prec UP: 50.5%
Prec DOWN: 0.0%
Binomial p-value: 1.0000

Model: RSI Heuristic Rule
Signals N: 666
Coverage: 100.0%
Val Acc: 50.5%
Test Acc: 50.5%
Wilson 95% CI: [46.7%, 54.2%]
Prec UP: 50.5%
Prec DOWN: 50.0%
Binomial p-value: 0.5155

Model: Logistic Regression
Signals N: 666
Coverage: 100.0%
Val Acc: 52.3%
Test Acc: 51.1%
Wilson 95% CI: [47.3%, 54.8%]
Prec UP: 51.3%
Prec DOWN: 50.7%
Binomial p-value: 0.3932

Model: Ablation (Indicators-Only, 17 Feat)
Signals N: 405
Coverage: 60.8%
Val Acc: 51.7%
Test Acc: 54.81%
Wilson 95% CI: [49.9%, 59.6%]
Prec UP: 55.6%
Prec DOWN: 53.5%
Binomial p-value: 0.0438

Model: XGBoost (Raw 0.50)
Signals N: 666
Coverage: 100.0%
Val Acc: 49.2%
Test Acc: 53.60%
Wilson 95% CI: [49.8%, 57.4%]
Prec UP: 53.3%
Prec DOWN: 54.0%
Binomial p-value: 0.0560

Model: XGBoost (Confident Filter)
Signals N: 409 (257 UP / 152 DOWN)
Coverage: 61.4%
Val Acc: 50.3%
Test Acc: 55.01%
Wilson 95% CI: [50.2%, 59.8%]
Prec UP: 53.3%
Prec DOWN: 57.9%
Binomial p-value: 0.0362

6.2 How I read these numbers
- On the held-out test set, the confident model emitted 409 signals and was correct on 225 of them (55.01% accuracy).
- The exact one-sided binomial test against the 50.5% baseline gave p = 0.0362. Because p < 0.05, the 4.5 percentage point advantage over baseline is statistically distinguishable from chance at the 95% confidence level.
- The 95% Wilson confidence range is 50.2% to 59.8%. The lower bound sits slightly above 50%, confirming a modest, realistic edge.
- Precision is balanced on both sides: 53.3% on UP calls and 57.9% on DOWN calls.

Table 6: Probability Calibration Bins

Bin [30% to 45%] (Bearish calls): Predicted Avg = 40.4%, Actual UP Rate = 42.8% (DOWN win rate = 57.2%), Count = 145
Bin [45% to 50%] (Neutral / Low confidence): Predicted Avg = 47.7%, Actual UP Rate = 51.4%, Count = 111
Bin [50% to 55%] (Neutral / Low confidence): Predicted Avg = 52.6%, Actual UP Rate = 53.4%, Count = 146
Bin [55% to 70%] (Bullish calls): Predicted Avg = 59.9%, Actual UP Rate = 53.0%, Count = 247

[INSERT SCREENSHOT: Calibration Curve and Probability Histogram (results/calibration_curve.png)]
Caption: Figure 2 shows the reliability curve and distribution of predicted probabilities.

6.3 Cost-aware paper test
I ran a paper backtest on the held-out test set deducting 2.0 basis points (0.02%) round-trip friction on every trade (1.0 bps half-spread plus 1.0 bps slippage):
- Initial balance: 100,000.00 dollars
- Strategy net return: +6.01% (Ending balance: 106,012.13 dollars)
- Buy and Hold TSLA return: -6.10% (Ending balance: 93,898.07 dollars)
- Net outperformance alpha: +12.11%
- Win rate after costs: 55.02% (230 Wins / 188 Losses out of 418 trades)
- Profit factor: 1.10
- Annualized Sharpe ratio: 4.72
- Maximum drawdown: -7.63% (compared to -18.03% for Buy and Hold)

[INSERT SCREENSHOT: Cost-Aware Backtest Equity Curve (results/equity_curve.png)]
Caption: Figure 3 shows the cumulative equity curve with 2.0 bps round-trip friction. The strategy preserved capital and gained +6.01% while Tesla fell -6.10%.

6.4 Live test results
During live market hours on November 16, 2026, the indicator ran continuously on live NYSE data. Across 27 logged candles, the live directional win rate was 55.6% (5 Wins / 4 Losses) on resolved trades, with remaining candles safely kept in HOLD or Pending status. The server ran with 100% uptime and zero crashes.

[INSERT SCREENSHOT: Live Dashboard during NYSE Market Hours]
Caption: Figure 4 shows the live web dashboard during active trading, displaying the current signal, confidence rating, TradingView candlesticks, and the win rate badge.

[INSERT SCREENSHOT: Live Prediction Log Table with Filter Pills and Verdicts]
Caption: Figure 5 shows the live prediction ledger recording time, signal, confidence, entry price, targets, actual price, and color-coded verdicts.


7. Error Analysis and Ablations

7.1 Where the model fails

Table 7: Accuracy by Time of Day (Held-Out Test Set)
Morning (9:30 to 11:30 ET): 55.8% accuracy (147 signals)
Midday (11:30 to 14:00 ET): 54.9% accuracy (144 signals)
Afternoon (14:00 to 16:00 ET): 54.2% accuracy (118 signals)

Table 8: Accuracy by Volume Ratio (Held-Out Test Set)
Low Volume (< 0.8x average): 53.8% accuracy (182 signals)
Normal Volume (0.8x to 1.5x average): 53.3% accuracy (184 signals)
High Volume (> 1.5x average): 67.4% accuracy (43 signals)

The error breakdown shows two clear findings:
1. High volume breakouts are the most reliable regime: when volume ratio exceeded 1.5x, accuracy surged to 67.4%.
2. The first 10 minutes of market open (9:30 to 9:40 AM ET) produce noisy signals due to wide opening bid-ask spreads on the IEX feed.

[INSERT SCREENSHOT: Confusion Matrices (results/confusion_matrices.png)]
Caption: Figure 6 shows the confusion matrices for Logistic Regression and XGBoost.

7.2 Feature importance
The top five features by Gini importance were:
1. stoch_k (11.8%): Measures price position within the 14-candle range.
2. rsi_14 (9.4%): Momentum balance between recent gains and losses.
3. price_vs_vwap (8.2%): Distance from session volume-weighted average price.
4. vol_ratio (7.6%): Confirms whether moves are backed by volume.
5. price_vs_bb_upper (6.9%): Detects upper volatility envelope expansion.

[INSERT SCREENSHOT: Feature Importance Bar Chart (results/feature_importance.png)]
Caption: Figure 7 displays the top 15 features used by the XGBoost model.

7.3 Ablation study (testing the main idea)

Table 9: Feature Ablation Comparison
Indicators-Only (17 Features): 54.81% accuracy, 405 signals, p = 0.0438
Full 35 Features (Indicators + Volume + Timing): 55.01% accuracy, 409 signals, p = 0.0362

The ablation study confirms that adding volume ratios and session timing provided a modest improvement: accuracy increased from 54.81% to 55.01%, signal count increased by 4, and the p-value improved from 0.0438 to 0.0362.

7.4 Stability checks

Table 10: Stability Across Random Seeds
Seed 42: 55.01% accuracy (409 signals)
Seed 100: 55.28% accuracy (407 signals)
Seed 2026: 54.50% accuracy (411 signals)
Mean: 54.93% plus or minus 0.40% across seeds.

Table 11: Threshold Sensitivity
52% / 48% thresholds: 55.16% accuracy (562 signals, 84.4% coverage)
55% / 45% thresholds: 55.01% accuracy (409 signals, 61.4% coverage)
58% / 42% thresholds: 54.75% accuracy (263 signals, 39.5% coverage)

Table 12: Training Window Size
60 days of training: 56.38% accuracy (486 signals)
120 days of training: 55.01% accuracy (409 signals)


8. The Live Indicator

8.1 What the trader sees
- Signal banner: clear UP, DOWN, or NEUTRAL badge.
- Calibrated confidence: exact probability reading (for example, "Down, 60.4%").
- Interactive chart: live 5-minute candlesticks with volume and signal markers via TradingView Lightweight Charts.
- Prediction log: table recording candle time, signal, confidence, entry price, targets, actual price, and color-coded verdicts (WIN, LOSS, HOLD, FLAT).

8.2 Guaranteed zero repainting
1. The server checks Alpaca every 60 seconds.
2. It ignores the active candle and evaluates only closed 5-minute bars.
3. Every signal is given an ID, timestamp, and entry price, and is written immediately to logs/prediction_log.json on disk.
4. Past signals are never edited, backfilled, or erased.

8.3 Volatility target heuristics
The dashboard displays Base, Bull, and Bear price levels based on ATR:
- Base Target: Current price plus or minus (ATR x 0.8).
- Bull Target: Current price plus (ATR x 1.5).
- Bear Target: Current price minus (ATR x 1.2) (stop-loss level).
Note: These targets are ATR-based volatility rules to help traders manage risk, not direct machine learning outputs.

8.4 When not to trust the indicator
1. Major macro news releases (Fed interest rate announcements, CPI data).
2. The first 10 minutes of regular market hours (9:30 to 9:40 AM ET).
3. Low-volume periods where volume ratio is below 0.4x.
4. When market data feeds experience network delays.


9. Limitations, Ethics and Conclusion

9.1 Limitations
- Tested on only one stock (Tesla); results may not generalize to lower-beta stocks.
- 120 days represents a single broad market regime.
- IEX volume represents a subset of total market volume.
- A 55% win rate means that about 45 out of every 100 trades will still lose, requiring disciplined stop-loss management.

9.2 Ethics and safe use
- Built strictly for academic learning; not intended as financial advice.
- Tested using paper trading with zero real capital risked.
- API keys are stored in an uncommitted .env file.
- The dashboard highlights invalidation levels to encourage risk-managed trading.

9.3 Conclusion
This project demonstrates that while predicting short-term stock direction is challenging, combining stationary technical indicators with volume and volatility features using gradient boosted trees can produce a small, statistically valid edge (55.01% accuracy, p = 0.0362). The critical component for profitability is selective confidence rejection: abstaining during low-confidence choppy periods allows the strategy to overcome transaction friction and protect capital (+6.01% net return after 2 bps costs).

9.4 Future work
- Extend training to multi-year datasets covering both bull and bear market cycles.
- Incorporate Level 2 limit order book depth features.
- Add market-wide ETF features (QQQ and SPY) to condition predictions on broader index momentum.


10. Reproducibility Notes

- Environment: Python 3.10+, dependencies in requirements.txt.
- Credentials: Create .env with ALPACA_API_KEY, ALPACA_SECRET_KEY, and TICKER=TSLA.
- Commands to reproduce:
  1. python src/step1_download_data.py
  2. python src/step2_explore_data.py
  3. python src/step4_features_and_target.py
  4. python src/step5_train_model.py
  5. python src/step6_scenarios.py
  6. python src/step8_benchmarks_and_audit.py
  7. python src/step9_cost_aware_backtest.py
  8. python src/step7_live_server.py (then open http://localhost:8000)
- Trained model file: models/tsla_xgb_model.pkl
- Persistent prediction log: logs/prediction_log.json


References

1. Chen, T., and Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785-794.
2. Fischer, T., and Krauss, C. (2018). Deep learning with long short-term memory networks for financial market predictions. European Journal of Operational Research, 270(2), 654-669.
3. Lopez de Prado, M. (2018). Advances in Financial Machine Learning. John Wiley and Sons, Hoboken, New Jersey.
4. Alpaca Markets. (2026). Alpaca Market Data API v2 Documentation. https://alpaca.markets/docs/
5. TradingView. (2026). Lightweight Charts Library Documentation. https://tradingview.github.io/lightweight-charts/
