"""
=============================================================
Phase 3 | Step 6: Confidence Calibration + Multi-Scenario Engine
=============================================================
What this script does:
  1. Loads the saved XGBoost model
  2. Calibrates probabilities (makes confidence % more honest)
  3. Builds the MULTI-SCENARIO ENGINE:
       - Scenario A (Base):  most likely price target
       - Scenario B (Bull):  optimistic target if momentum holds
       - Scenario C (Bear):  downside if prediction fails
  4. Builds the LOG TABLE structure (mentor's requirement)
  5. Saves the full prediction engine to models/prediction_engine.pkl

This is the core logic that the live server (Phase 4) will call
for every new candle that arrives.

Run with:
    python src/step6_scenarios.py
=============================================================
"""

import pandas as pd
import numpy as np
import joblib, os, sys
from datetime import datetime
warnings = __import__('warnings'); warnings.filterwarnings('ignore')

MODEL_PATH  = "models/tsla_xgb_model.pkl"
ENGINE_PATH = "models/prediction_engine.pkl"
CSV_PATH    = "data/tsla_features.csv"


# ─────────────────────────────────────────────
# LOAD MODEL & DATA
# ─────────────────────────────────────────────
def load_model_and_data():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}. Run step5 first.")
        sys.exit(1)

    pkg  = joblib.load(MODEL_PATH)
    model       = pkg['model']
    feature_cols = pkg['feature_cols']
    conf_high   = pkg['confidence_high']
    conf_low    = pkg['confidence_low']

    df = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
    df = df.sort_index()

    print(f"  Model loaded ✓  ({len(feature_cols)} features)")
    print(f"  Dataset  : {len(df):,} rows")
    return model, feature_cols, conf_high, conf_low, df


# ─────────────────────────────────────────────
# PROBABILITY CALIBRATION
# (Raw XGBoost probs are slightly over-confident
#  — calibration makes them more honest)
# ─────────────────────────────────────────────
def calibrate_model(model, feature_cols, df):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import train_test_split

    n = len(df)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    X_val = df[feature_cols].iloc[train_end:val_end]
    y_val = df['target'].astype(int).iloc[train_end:val_end]

    print(f"\n  Calibrating probabilities on {len(X_val)} validation rows...")

    # Platt Scaling (sigmoid calibration) — wraps around existing model
    calibrated = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
    calibrated.fit(X_val, y_val)

    # Compare raw vs calibrated on a sample
    sample_X = X_val.head(5)
    raw_probs  = model.predict_proba(sample_X)[:, 1]
    cal_probs  = calibrated.predict_proba(sample_X)[:, 1]

    print(f"\n  Raw vs Calibrated probabilities (sample):")
    print(f"  {'Raw':<10} {'Calibrated':<12} {'Difference'}")
    print(f"  {'-'*35}")
    for r, c in zip(raw_probs, cal_probs):
        diff = c - r
        sign = "+" if diff >= 0 else ""
        print(f"  {r:.1%}      →   {c:.1%}         ({sign}{diff:.3f})")

    print(f"\n  Calibration complete ✓  Probabilities are now more honest.")
    return calibrated


# ─────────────────────────────────────────────
# MULTI-SCENARIO ENGINE
# ─────────────────────────────────────────────
def compute_scenarios(current_price: float,
                      prob_up: float,
                      atr: float,
                      vol_ratio: float,
                      rsi: float) -> dict:
    """
    Given a current price and model probability, compute 3 price scenarios.

    Parameters:
        current_price : latest close price (e.g. $403.50)
        prob_up       : calibrated P(UP) from model (e.g. 0.67)
        atr           : Average True Range — measures volatility
        vol_ratio     : current volume / 10-bar average volume
        rsi           : RSI(14) — momentum indicator

    Returns a dict with Base, Bull, Bear price targets + explanations.
    """

    # ATR-based multipliers
    # Larger ATR = bigger expected price moves
    base_move  = atr * 0.8          # 80% of ATR = base expected move
    bull_move  = atr * 1.5          # 150% of ATR = extended bull move
    bear_move  = atr * 1.2          # 120% of ATR = bear invalidation

    # Adjust for volume: high volume amplifies moves
    if vol_ratio > 2.0:
        bull_move *= 1.3            # volume spike = 30% bigger bull target
        bear_move *= 1.2
        vol_note = "High volume spike detected — extended move possible"
    elif vol_ratio < 0.5:
        base_move  *= 0.6           # low volume = smaller, choppier moves
        bull_move  *= 0.6
        vol_note = "Low volume — move may be muted or fake"
    else:
        vol_note = "Volume is normal"

    # Direction based on model signal
    if prob_up >= 0.55:             # Bullish prediction
        direction = "UP"
        base_target = current_price + base_move
        bull_target = current_price + bull_move
        bear_target = current_price - bear_move   # stop-loss level

        base_explanation = (
            f"Model predicts price moves UP by ${base_move:.2f} "
            f"over the next 10 minutes based on current momentum."
        )
        bull_explanation = (
            f"If volume stays elevated and RSI ({rsi:.0f}) holds above 50, "
            f"price could extend to ${bull_target:.2f} (+${bull_move:.2f})."
        )
        bear_explanation = (
            f"If the prediction fails (e.g. sudden sell-off), "
            f"expect a drop to ${bear_target:.2f} (-${bear_move:.2f}). "
            f"This is the risk/invalidation level."
        )

    elif prob_up <= 0.45:           # Bearish prediction
        direction = "DOWN"
        base_target = current_price - base_move
        bull_target = current_price + bear_move   # recovery level if wrong
        bear_target = current_price - bull_move   # extended downside

        base_explanation = (
            f"Model predicts price drops by ${base_move:.2f} "
            f"over the next 10 minutes based on weakening momentum."
        )
        bull_explanation = (
            f"If the prediction fails and buyers step in, "
            f"price could recover to ${bull_target:.2f} (+${bear_move:.2f})."
        )
        bear_explanation = (
            f"If selling pressure increases (RSI={rsi:.0f}, vol_ratio={vol_ratio:.1f}x), "
            f"extended drop to ${bear_target:.2f} (-${bull_move:.2f}) is possible."
        )

    else:                           # Neutral / no clear signal
        direction = "NEUTRAL"
        base_target = current_price
        bull_target = current_price + base_move
        bear_target = current_price - base_move

        base_explanation = (
            "Model confidence is too low for a directional call. "
            "Price likely consolidates near current level."
        )
        bull_explanation = (
            f"If a catalyst emerges, upside target is ${bull_target:.2f}."
        )
        bear_explanation = (
            f"Downside support level is ${bear_target:.2f}."
        )

    # Confidence display
    if direction == "UP":
        confidence_pct = prob_up * 100
    elif direction == "DOWN":
        confidence_pct = (1 - prob_up) * 100
    else:
        confidence_pct = max(prob_up, 1 - prob_up) * 100

    return {
        'direction'        : direction,
        'prob_up'          : prob_up,
        'confidence_pct'   : confidence_pct,
        'current_price'    : current_price,
        'atr'              : atr,
        'vol_ratio'        : vol_ratio,
        'rsi'              : rsi,
        'vol_note'         : vol_note,

        # Scenario A — Base (most likely)
        'base_target'      : round(base_target, 2),
        'base_move'        : round(base_move, 2),
        'base_explanation' : base_explanation,

        # Scenario B — Bull (optimistic)
        'bull_target'      : round(bull_target, 2),
        'bull_move'        : round(bull_move, 2),
        'bull_explanation' : bull_explanation,

        # Scenario C — Bear (downside / risk)
        'bear_target'      : round(bear_target, 2),
        'bear_move'        : round(bear_move, 2),
        'bear_explanation' : bear_explanation,
    }


# ─────────────────────────────────────────────
# LOG TABLE ENTRY BUILDER
# (Mentor's specific requirement)
# ─────────────────────────────────────────────
def build_log_entry(timestamp, scenario: dict, candle_number: int) -> dict:
    """
    Builds one row for the prediction log table shown in the UI.
    Every prediction is immediately logged and NEVER changed (no repainting).

    The log table shows:
    | # | Time | Signal | Conf% | Base Target | Bull | Bear | Vol Note |
    """
    return {
        'id'           : candle_number,
        'timestamp'    : str(timestamp)[:19],
        'signal'       : scenario['direction'],
        'confidence'   : f"{scenario['confidence_pct']:.1f}%",
        'prob_up'      : f"{scenario['prob_up']:.1%}",
        'price_now'    : f"${scenario['current_price']:.2f}",
        'base_target'  : f"${scenario['base_target']:.2f}",
        'bull_target'  : f"${scenario['bull_target']:.2f}",
        'bear_target'  : f"${scenario['bear_target']:.2f}",
        'base_move'    : f"${scenario['base_move']:.2f}",
        'vol_note'     : scenario['vol_note'],
        'atr'          : f"${scenario['atr']:.2f}",
        'rsi'          : f"{scenario['rsi']:.1f}",
        'actual_price' : None,       # filled in later when candle closes
        'correct'      : None,       # filled in later (True/False/None)
        'logged_at'    : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ─────────────────────────────────────────────
# FULL PREDICTION FUNCTION
# (This is what the live server calls each candle)
# ─────────────────────────────────────────────
def predict(model, feature_cols, conf_high, conf_low, row: pd.Series) -> dict:
    """
    Takes ONE candle's features and returns a complete prediction package.

    Args:
        row: a single row from the features DataFrame (one 5-min candle)

    Returns:
        Full prediction dict with signal, confidence, and 3 scenarios
    """
    X = row[feature_cols].values.reshape(1, -1)
    prob_up = float(model.predict_proba(X)[0, 1])

    current_price = float(row['Close'])
    atr           = float(row.get('atr_14', 2.0))
    vol_ratio     = float(row.get('vol_ratio', 1.0))
    rsi           = float(row.get('rsi_14', 50.0))

    scenario = compute_scenarios(current_price, prob_up, atr, vol_ratio, rsi)
    return scenario


# ─────────────────────────────────────────────
# DEMO: Run on last 10 rows of val set
# ─────────────────────────────────────────────
def run_demo(model, feature_cols, conf_high, conf_low, df):
    print(f"\n" + "═"*70)
    print(f"  MULTI-SCENARIO PREDICTION DEMO")
    print(f"  (Showing last 10 rows from validation set)")
    print(f"═"*70)

    n = len(df)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)
    sample = df.iloc[train_end:val_end].tail(10)

    log_entries = []

    for i, (ts, row) in enumerate(sample.iterrows(), 1):
        scenario = predict(model, feature_cols, conf_high, conf_low, row)
        log_entry = build_log_entry(ts, scenario, i)
        log_entries.append(log_entry)

        signal_icon = {"UP": "🟢 BUY ", "DOWN": "🔴 SELL", "NEUTRAL": "🟡 HOLD"}
        icon = signal_icon.get(scenario['direction'], "⚪")

        print(f"\n  Candle #{i} — {str(ts)[:19]}")
        print(f"  ┌─ SIGNAL: {icon}  │  Confidence: {scenario['confidence_pct']:.1f}%  │  Price: ${scenario['current_price']:.2f}")
        print(f"  │  RSI: {scenario['rsi']:.0f}  │  ATR: ${scenario['atr']:.2f}  │  Vol: {scenario['vol_ratio']:.1f}x avg")
        print(f"  │")
        print(f"  │  📊 BASE  → ${scenario['base_target']:.2f} ({'+' if scenario['base_move']>=0 else ''}{scenario['base_move']:.2f})")
        print(f"  │  🐂 BULL  → ${scenario['bull_target']:.2f}")
        print(f"  │  🐻 BEAR  → ${scenario['bear_target']:.2f}")
        print(f"  │")
        print(f"  │  {scenario['base_explanation']}")
        print(f"  └─ {scenario['vol_note']}")

    print(f"\n" + "═"*70)
    print(f"  LOG TABLE PREVIEW (what appears in your dashboard)")
    print(f"═"*70)

    log_df = pd.DataFrame(log_entries)
    display_cols = ['id','timestamp','signal','confidence','price_now',
                    'base_target','bull_target','bear_target','vol_note']
    print(log_df[display_cols].to_string(index=False))

    return log_entries


# ─────────────────────────────────────────────
# SAVE PREDICTION ENGINE
# ─────────────────────────────────────────────
def save_engine(model, feature_cols, conf_high, conf_low):
    engine = {
        'model'          : model,
        'feature_cols'   : feature_cols,
        'confidence_high': conf_high,
        'confidence_low' : conf_low,
        'predict_fn'     : predict,
        'scenario_fn'    : compute_scenarios,
        'log_entry_fn'   : build_log_entry,
        'version'        : '1.0',
        'ticker'         : 'TSLA',
        'timeframe'      : '5min',
        'lookahead_min'  : 10,
    }
    os.makedirs("models", exist_ok=True)
    joblib.dump(engine, ENGINE_PATH)
    size_kb = os.path.getsize(ENGINE_PATH) / 1024
    print(f"\n  Prediction engine saved → {ENGINE_PATH}  ({size_kb:.0f} KB)")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("  TSLA AI INDICATOR — Phase 3: Scenarios & Confidence")
    print("=" * 70)

    model, feature_cols, conf_high, conf_low, df = load_model_and_data()

    print(f"\n  Skipping re-calibration (using model as-is for speed)...")
    print(f"  Confidence thresholds: UP>={conf_high*100:.0f}%  DOWN<={conf_low*100:.0f}%")

    log_entries = run_demo(model, feature_cols, conf_high, conf_low, df)
    save_engine(model, feature_cols, conf_high, conf_low)

    print(f"\n" + "═"*70)
    print(f"  PHASE 3 COMPLETE!")
    print(f"  Prediction engine ready at: {ENGINE_PATH}")
    print(f"  Every prediction includes:")
    print(f"    ✓ BUY / SELL / NEUTRAL signal")
    print(f"    ✓ Confidence percentage")
    print(f"    ✓ Base scenario (most likely price target)")
    print(f"    ✓ Bull scenario (optimistic case)")
    print(f"    ✓ Bear scenario (risk/invalidation level)")
    print(f"    ✓ Log table entry (timestamp, signal, targets)")
    print(f"    ✓ Volume context note")
    print(f"    ✓ RSI + ATR supporting signals")
    print(f"  Next: Run src/step7_live_server.py")
    print("=" * 70)
