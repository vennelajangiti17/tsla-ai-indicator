"""
=============================================================
Phase 2 | Step 5: Train XGBoost Model with Walk-Forward Validation
=============================================================
What this script does:
  1. Loads the ML-ready feature dataset from Phase 1
  2. Runs WALK-FORWARD VALIDATION (5 folds, no data leakage)
  3. Trains a final XGBoost model on the full training set
  4. Applies a CONFIDENCE FILTER (only predict when sure)
  5. Reports full metrics: accuracy, precision, recall, F1
  6. Shows which features matter most (feature importance)
  7. Saves the trained model to models/tsla_xgb_model.pkl

Run with:
    python src/step5_train_model.py
=============================================================
"""

import pandas as pd
import numpy as np
import os, sys, joblib, warnings
warnings.filterwarnings('ignore')

CSV_PATH   = "data/tsla_features.csv"
MODEL_PATH = "models/tsla_xgb_model.pkl"

# ── Confidence filter threshold ───────────────────────────
# Only issue a signal when the model is THIS sure (or more)
# Rows between LOW and HIGH are labelled "NEUTRAL / NO SIGNAL"
CONFIDENCE_HIGH = 0.55   # predict UP   when P(UP) >= 55%
CONFIDENCE_LOW  = 0.45   # predict DOWN when P(UP) <= 45%

# ── Walk-forward folds ────────────────────────────────────
N_FOLDS = 5   # split training data into 5 chronological folds

# ── Feature columns (must match step4 output) ─────────────
DROP_COLS = ['Open','High','Low','Close','Volume',
             'future_close','future_pct_change','target',
             'typical_price']


# ─────────────────────────────────────────────
# LOAD & SPLIT DATA
# ─────────────────────────────────────────────
def load_and_split():
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: '{CSV_PATH}' not found. Run step4 first.")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
    df = df.sort_index()

    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feature_cols]
    y = df['target'].astype(int)

    n = len(df)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    X_val   = X.iloc[train_end:val_end]
    y_val   = y.iloc[train_end:val_end]
    X_test  = X.iloc[val_end:]
    y_test  = y.iloc[val_end:]

    print(f"  Loaded {len(df):,} rows, {len(feature_cols)} features")
    print(f"  TRAIN : {len(X_train):,} rows")
    print(f"  VAL   : {len(X_val):,} rows")
    print(f"  TEST  : {len(X_test):,} rows  (LOCKED — only used at the end)")

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


# ─────────────────────────────────────────────
# WALK-FORWARD VALIDATION
# (The honest way to validate on time-series)
# ─────────────────────────────────────────────
def walk_forward_validation(X_train, y_train):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("ERROR: xgboost not installed. Run: pip install xgboost")
        sys.exit(1)

    print(f"\n" + "═"*58)
    print(f"  WALK-FORWARD VALIDATION  ({N_FOLDS} folds)")
    print(f"═"*58)
    print(f"  Each fold: train on past → test on immediate future")
    print(f"  This mirrors how the model runs in live trading\n")

    n = len(X_train)
    fold_size = n // (N_FOLDS + 1)

    fold_accuracies     = []
    fold_conf_accuracies = []
    fold_coverage       = []

    for fold in range(1, N_FOLDS + 1):
        # Each fold uses all previous data to train, next chunk to validate
        train_end = fold * fold_size
        val_start = train_end
        val_end   = val_start + fold_size

        if val_end > n:
            break

        X_f_train = X_train.iloc[:train_end]
        y_f_train = y_train.iloc[:train_end]
        X_f_val   = X_train.iloc[val_start:val_end]
        y_f_val   = y_train.iloc[val_start:val_end]

        # Train a quick model for this fold
        model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            verbosity=0
        )
        model.fit(X_f_train, y_f_train)

        # Raw accuracy (all predictions)
        probs  = model.predict_proba(X_f_val)[:, 1]
        preds  = (probs >= 0.5).astype(int)
        raw_acc = (preds == y_f_val.values).mean()

        # Confidence-filtered accuracy
        confident_mask = (probs >= CONFIDENCE_HIGH) | (probs <= CONFIDENCE_LOW)
        coverage = confident_mask.mean()

        if confident_mask.sum() > 0:
            conf_preds  = (probs[confident_mask] >= 0.5).astype(int)
            conf_actual = y_f_val.values[confident_mask]
            conf_acc    = (conf_preds == conf_actual).mean()
        else:
            conf_acc = 0.0

        fold_accuracies.append(raw_acc)
        fold_conf_accuracies.append(conf_acc)
        fold_coverage.append(coverage)

        bar = "█" * int(raw_acc * 20) + "░" * (20 - int(raw_acc * 20))
        print(f"  Fold {fold}: [{bar}] "
              f"Raw={raw_acc*100:.1f}%  "
              f"Confident={conf_acc*100:.1f}%  "
              f"Coverage={coverage*100:.0f}%")

    avg_raw  = np.mean(fold_accuracies)
    avg_conf = np.mean(fold_conf_accuracies)
    avg_cov  = np.mean(fold_coverage)

    print(f"\n  Average across {N_FOLDS} folds:")
    print(f"    Raw accuracy        : {avg_raw*100:.1f}%")
    print(f"    Confident accuracy  : {avg_conf*100:.1f}%  ← use this number in your report")
    print(f"    Signal coverage     : {avg_cov*100:.0f}%  of candles get a signal")

    return avg_raw, avg_conf


# ─────────────────────────────────────────────
# TRAIN FINAL MODEL
# ─────────────────────────────────────────────
def train_final_model(X_train, y_train, X_val, y_val):
    from xgboost import XGBClassifier

    print(f"\n" + "═"*58)
    print(f"  TRAINING FINAL MODEL on full training set...")
    print(f"═"*58)

    model = XGBClassifier(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=10,
        gamma=0.5,
        reg_alpha=0.5,
        reg_lambda=2.0,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        verbosity=0
    )

    # Train without early stopping — fixed number of trees
    model.fit(X_train, y_train)
    print(f"  Trained {model.n_estimators} trees  ✓")

    return model


# ─────────────────────────────────────────────
# EVALUATE ON VALIDATION SET
# ─────────────────────────────────────────────
def evaluate(model, X, y, label="Validation"):
    from sklearn.metrics import (accuracy_score, precision_score,
                                  recall_score, f1_score, confusion_matrix)

    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)

    acc  = accuracy_score(y, preds)
    prec = precision_score(y, preds, zero_division=0)
    rec  = recall_score(y, preds, zero_division=0)
    f1   = f1_score(y, preds, zero_division=0)
    cm   = confusion_matrix(y, preds)

    # Confidence-filtered metrics
    conf_mask = (probs >= CONFIDENCE_HIGH) | (probs <= CONFIDENCE_LOW)
    coverage  = conf_mask.mean()

    if conf_mask.sum() > 0:
        conf_preds  = (probs[conf_mask] >= 0.5).astype(int)
        conf_acc    = accuracy_score(y.values[conf_mask], conf_preds)
        conf_prec   = precision_score(y.values[conf_mask], conf_preds, zero_division=0)
        conf_rec    = recall_score(y.values[conf_mask], conf_preds, zero_division=0)
        n_up        = (conf_preds == 1).sum()
        n_down      = (conf_preds == 0).sum()
        n_neutral   = (~conf_mask).sum()
    else:
        conf_acc = conf_prec = conf_rec = 0.0
        n_up = n_down = n_neutral = 0

    print(f"\n" + "═"*58)
    print(f"  {label.upper()} SET RESULTS")
    print(f"═"*58)

    print(f"\n  ── All Predictions (no filter) ──")
    print(f"  Accuracy  : {acc*100:.1f}%")
    print(f"  Precision : {prec*100:.1f}%  (of UP predictions, how many were right?)")
    print(f"  Recall    : {rec*100:.1f}%  (of actual UPs, how many did we catch?)")
    print(f"  F1 Score  : {f1*100:.1f}%")

    print(f"\n  ── Confusion Matrix ──")
    print(f"                 Predicted DOWN  Predicted UP")
    print(f"  Actual DOWN :      {cm[0][0]:>5}            {cm[0][1]:>5}")
    print(f"  Actual UP   :      {cm[1][0]:>5}            {cm[1][1]:>5}")

    print(f"\n  ── With Confidence Filter (>={CONFIDENCE_HIGH*100:.0f}% or <={CONFIDENCE_LOW*100:.0f}%) ──")
    print(f"  Confident accuracy : {conf_acc*100:.1f}%  ← THIS is the number that matters for live trading")
    print(f"  Confident precision: {conf_prec*100:.1f}%")
    print(f"  Signal coverage    : {coverage*100:.0f}%  ({n_up} UP + {n_down} DOWN + {n_neutral} NEUTRAL signals)")

    return {
        'raw_accuracy': acc,
        'conf_accuracy': conf_acc,
        'coverage': coverage,
        'precision': prec,
        'recall': rec,
        'f1': f1
    }


# ─────────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────
def print_feature_importance(model, feature_cols):
    print(f"\n" + "═"*58)
    print(f"  TOP 15 MOST IMPORTANT FEATURES")
    print(f"  (What the model relies on most to make predictions)")
    print(f"═"*58)

    importance = pd.Series(
        model.feature_importances_,
        index=feature_cols
    ).sort_values(ascending=False)

    for i, (feat, score) in enumerate(importance.head(15).items()):
        bar = "█" * int(score * 300)
        print(f"  {i+1:>2}. {feat:<25} {bar}  ({score:.4f})")

    print(f"\n  Insight: Features at the top are what your model")
    print(f"  relies on most. Include these in your report!")


# ─────────────────────────────────────────────
# SAVE MODEL
# ─────────────────────────────────────────────
def save_model(model, feature_cols):
    os.makedirs("models", exist_ok=True)

    # Save model + feature list together (critical for live inference)
    model_package = {
        'model': model,
        'feature_cols': feature_cols,
        'confidence_high': CONFIDENCE_HIGH,
        'confidence_low': CONFIDENCE_LOW,
        'lookahead_candles': 3,
        'timeframe_minutes': 15
    }

    joblib.dump(model_package, MODEL_PATH)
    size_kb = os.path.getsize(MODEL_PATH) / 1024
    print(f"\n  Model saved to : {MODEL_PATH}  ({size_kb:.0f} KB)")
    print(f"  Feature list saved inside model package ✓")
    print(f"  Confidence thresholds saved inside model package ✓")


# ─────────────────────────────────────────────
# QUICK PREDICTION DEMO
# ─────────────────────────────────────────────
def demo_prediction(model, X_val, feature_cols):
    print(f"\n" + "═"*58)
    print(f"  LIVE PREDICTION DEMO (last 5 val rows)")
    print(f"  (This is what the live server will produce per candle)")
    print(f"═"*58)
    print(f"  {'Timestamp':<26} {'P(UP)':>6}  {'Signal':<10}  Confidence")
    print(f"  {'-'*54}")

    sample = X_val.tail(5)
    probs  = model.predict_proba(sample)[:, 1]

    for ts, prob in zip(sample.index, probs):
        if prob >= CONFIDENCE_HIGH:
            signal = "🟢 BUY"
            conf   = f"{prob*100:.1f}%"
        elif prob <= CONFIDENCE_LOW:
            signal = "🔴 SELL"
            conf   = f"{(1-prob)*100:.1f}%"
        else:
            signal = "🟡 NEUTRAL"
            conf   = f"Low ({prob*100:.1f}%)"

        ts_str = str(ts)[:19]
        print(f"  {ts_str:<26} {prob:>5.1%}  {signal:<12}  {conf}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 58)
    print("  TSLA AI INDICATOR — Phase 2: Train Model")
    print("=" * 58)

    # 1. Load data
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = load_and_split()

    # 2. Walk-forward validation (honest evaluation)
    avg_raw, avg_conf = walk_forward_validation(X_train, y_train)

    # 3. Train final model
    model = train_final_model(X_train, y_train, X_val, y_val)

    # 4. Evaluate on validation set
    val_metrics = evaluate(model, X_val, y_val, label="Validation")

    # 5. PEEK at test set (just once — to report in your submission)
    print(f"\n  Unlocking TEST SET for final honest evaluation...")
    test_metrics = evaluate(model, X_test, y_test, label="Test (Out-of-Sample)")

    # 6. Feature importance
    print_feature_importance(model, feature_cols)

    # 7. Demo
    demo_prediction(model, X_val, feature_cols)

    # 8. Save
    save_model(model, feature_cols)

    # 9. Final summary
    print(f"\n" + "═"*58)
    print(f"  PHASE 2 COMPLETE — MODEL SUMMARY")
    print(f"═"*58)
    print(f"  Walk-forward confident accuracy : {avg_conf*100:.1f}%")
    print(f"  Validation confident accuracy   : {val_metrics['conf_accuracy']*100:.1f}%")
    print(f"  Test confident accuracy         : {test_metrics['conf_accuracy']*100:.1f}%")
    print(f"  Signal coverage                 : {val_metrics['coverage']*100:.0f}% of candles")
    print(f"\n  Baseline to beat : 55.0%")

    if val_metrics['conf_accuracy'] >= 0.55:
        print(f"  Result : ✅  PASSED — model beats the baseline!")
    else:
        print(f"  Result : ⚠️  Below baseline — we will tune hyperparameters")

    print(f"\n  Model saved → {MODEL_PATH}")
    print(f"  Next step  → python src/step6_scenarios.py")
    print("=" * 58)
