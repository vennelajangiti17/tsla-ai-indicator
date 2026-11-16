# Fair Comparison & Benchmark Table
*Evaluated on the exact same Chronological Split (Train 70% / Val 15% / Test 15%)*
*Mandated by IMPACT Lab Guidelines (Step 7 & Step 10)*

| Model | Model Type | Val Acc (%) | Test Acc (%) | Test Prec (UP) | Test Recall | Test F1 | Train Time | Size |
|-------|------------|-------------|--------------|----------------|-------------|---------|------------|------|
| Majority Class Baseline (Always UP) | Trivial Baseline | 52.0% | 51.3% | **51.3%** | 100.0% | 67.8% | 0.0s | 0.1 KB |
| RSI Heuristic Rule (RSI < 35 / > 65) | Domain Rule Baseline | 50.4% | 52.1% | **51.9%** | 88.8% | 65.5% | 0.0002s | 0.2 KB |
| Logistic Regression (Standardized) | Conventional ML | 50.1% | 49.0% | **50.3%** | 57.5% | 53.6% | 0.012s | 2.5 KB |
| Proposed XGBoost (Raw Threshold 0.50) | Gradient Boosted Trees | 53.2% | 53.2% | **54.0%** | 60.6% | 57.1% | 0.24s | 340.0 KB |
| Proposed XGBoost + Confidence Filter (P≥0.52/P≤0.48) | Filtered High-Confidence Signals | 54.2% | 53.8% | **55.0%** | 62.7% | 58.6% | 0.24s | 340.0 KB |

### Key Findings:
1. **Credible Baseline Comparison**: The Majority Class ("Always UP") achieves 51.3% on test.
2. **Superior Precision**: When high confidence is required ($P \ge 0.52$), **XGBoost achieves 55.0% precision on UP signals** on unseen test data, outperforming both naive guessing and Logistic Regression.
3. **Trader Value**: Filtering ambiguous chop protects trading capital from churn.
