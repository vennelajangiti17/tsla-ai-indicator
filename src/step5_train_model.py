"""
=============================================================
Phase 2 | Step 5: Train XGBoost Model with Walk-Forward Validation
=============================================================
Major Improvements Addressing Quant Review:
  1. 5-Fold Walk-Forward Cross Validation:
     - Honest expanding-window protocol.
     - Saves full metrics to results/walk_forward_metrics.csv and .json.
  2. Validation-Based Threshold Selection:
     - Confident threshold is chosen on VALIDATION set, NOT test.
  3. Single-Pass Unbiased Test Evaluation:
     - Held-out test set evaluated ONCE with frozen parameters.
  4. Metadata Consistency:
     - Package metadata explicitly stores lookahead_candles = 2 (10 min).
=============================================================
"""

import os, sys, json, joblib, warnings
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

warnings.filterwarnings('ignore')

CSV_PATH    = "data/tsla_features.csv"
MODEL_PATH  = "models/tsla_xgb_model.pkl"
RESULTS_DIR = "results"
os.makedirs("models", exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

N_FOLDS   = 5
DROP_COLS = [
    'Open','High','Low','Close','Volume',
    'future_close','future_time','future_pct_change','target'
]

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

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val     = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test   = X.iloc[val_end:], y.iloc[val_end:]

    print(f"  Loaded {len(df):,} rows, {len(feature_cols)} stationary features")
    print(f"  TRAIN : {len(X_train):,} rows ({str(X_train.index.min())[:10]} to {str(X_train.index.max())[:10]})")
    print(f"  VAL   : {len(X_val):,} rows ({str(X_val.index.min())[:10]} to {str(X_val.index.max())[:10]})")
    print(f"  TEST  : {len(X_test):,} rows ({str(X_test.index.min())[:10]} to {str(X_test.index.max())[:10]}) [LOCKED]")

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols

def run_walk_forward(X_train, y_train):
    print(f"\n" + "=" * 58)
    print(f"  WALK-FORWARD VALIDATION ({N_FOLDS} Folds)")
    print("=" * 58)
    print("  Rolling-origin evaluation simulating live sequential trading...\n")

    n = len(X_train)
    fold_size = n // (N_FOLDS + 1)
    fold_records = []

    for fold in range(1, N_FOLDS + 1):
        train_end = fold * fold_size
        val_start = train_end
        val_end   = min(val_start + fold_size, n)

        X_f_train = X_train.iloc[:train_end]
        y_f_train = y_train.iloc[:train_end]
        X_f_val   = X_train.iloc[val_start:val_end]
        y_f_val   = y_train.iloc[val_start:val_end]

        fold_model = XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=15,
            gamma=0.5,
            reg_alpha=0.5,
            reg_lambda=2.0,
            eval_metric='logloss',
            random_state=42,
            verbosity=0
        )
        fold_model.fit(X_f_train, y_f_train)

        probs = fold_model.predict_proba(X_f_val)[:, 1]
        preds = (probs >= 0.5).astype(int)
        raw_acc = accuracy_score(y_f_val, preds)

        # 54% confidence threshold
        conf_mask = (probs >= 0.54) | (probs <= 0.46)
        cov = conf_mask.mean()
        if conf_mask.sum() > 0:
            conf_preds = (probs[conf_mask] >= 0.5).astype(int)
            conf_acc = accuracy_score(y_f_val[conf_mask], conf_preds)
            up_mask = probs >= 0.54
            conf_prec_up = precision_score(y_f_val[up_mask], (probs[up_mask] >= 0.5).astype(int), zero_division=0) if up_mask.sum() > 0 else 0.0
        else:
            conf_acc = conf_prec_up = 0.0

        fold_records.append({
            'fold': fold,
            'train_samples': len(X_f_train),
            'test_samples': len(X_f_val),
            'raw_accuracy': round(raw_acc * 100, 2),
            'confident_accuracy': round(conf_acc * 100, 2),
            'precision_up': round(conf_prec_up * 100, 2),
            'signal_coverage': round(cov * 100, 1)
        })

        bar = "█" * int(raw_acc * 20) + "░" * (20 - int(raw_acc * 20))
        print(f"  Fold {fold}: [{bar}] Raw={raw_acc*100:.1f}%  Confident={conf_acc*100:.1f}%  UP Prec={conf_prec_up*100:.1f}%  Cov={cov*100:.0f}%")

    wf_df = pd.DataFrame(fold_records)
    wf_df.to_csv(os.path.join(RESULTS_DIR, "walk_forward_metrics.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, "walk_forward_metrics.json"), "w") as f:
        json.dump(fold_records, f, indent=2)

    avg_raw = wf_df['raw_accuracy'].mean()
    avg_conf = wf_df['confident_accuracy'].mean()
    print(f"\n  Average Walk-Forward Accuracy : {avg_raw:.2f}% (Raw) | {avg_conf:.2f}% (Confident)")
    print(f"  Saved walk-forward metrics to {RESULTS_DIR}/walk_forward_metrics.csv")
    return avg_raw, avg_conf

def train_and_tune(X_train, y_train, X_val, y_val):
    print(f"\n" + "=" * 58)
    print("  TRAINING FINAL MODEL & TUNING THRESHOLD ON VALIDATION SET")
    print("=" * 58)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=15,
        gamma=0.5,
        reg_alpha=0.5,
        reg_lambda=2.0,
        eval_metric='logloss',
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train)

    # Threshold selection strictly on VALIDATION set
    val_probs = model.predict_proba(X_val)[:, 1]
    best_thresh = 0.54
    best_val_prec = 0.0

    print("  Validation Set Threshold Tuning:")
    for thresh in np.arange(0.52, 0.58, 0.01):
        up_mask = val_probs >= thresh
        if up_mask.sum() >= 20: # ensure minimum statistical sample
            prec = precision_score(y_val[up_mask], (val_probs[up_mask] >= 0.5).astype(int), zero_division=0)
            cov = ((val_probs >= thresh) | (val_probs <= (1 - thresh))).mean()
            print(f"    Thresh {thresh:.2f}: Val UP Precision = {prec*100:.1f}%, Coverage = {cov*100:.1f}%")
            if prec > best_val_prec and cov >= 0.20:
                best_val_prec = prec
                best_thresh = thresh

    print(f"\n  Selected Optimal Threshold: {best_thresh:.2f} (Val Precision UP = {best_val_prec*100:.1f}%)")
    return model, best_thresh

def evaluate(model, X, y, thresh, label="Validation"):
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)

    raw_acc  = accuracy_score(y, preds)
    raw_prec = precision_score(y, preds, zero_division=0)
    raw_rec  = recall_score(y, preds, zero_division=0)
    raw_f1   = f1_score(y, preds, zero_division=0)
    cm       = confusion_matrix(y, preds)

    conf_mask = (probs >= thresh) | (probs <= (1 - thresh))
    cov = conf_mask.mean()
    if conf_mask.sum() > 0:
        conf_preds = (probs[conf_mask] >= 0.5).astype(int)
        conf_acc  = accuracy_score(y[conf_mask], conf_preds)
        up_mask   = probs >= thresh
        conf_prec_up = precision_score(y[up_mask], (probs[up_mask] >= 0.5).astype(int), zero_division=0) if up_mask.sum() > 0 else 0.0
        down_mask = probs <= (1 - thresh)
        conf_prec_down = precision_score(y[down_mask] == 0, (probs[down_mask] < 0.5), zero_division=0) if down_mask.sum() > 0 else 0.0
    else:
        conf_acc = conf_prec_up = conf_prec_down = 0.0

    print(f"\n" + "-" * 58)
    print(f"  {label.upper()} RESULTS (Threshold: {thresh:.2f})")
    print("-" * 58)
    print(f"  Raw Accuracy        : {raw_acc*100:.1f}%")
    print(f"  Raw Precision (UP)  : {raw_prec*100:.1f}%")
    print(f"  Raw F1 Score        : {raw_f1*100:.1f}%")
    print(f"  Confusion Matrix    : TN={cm[0][0]}, FP={cm[0][1]} | FN={cm[1][0]}, TP={cm[1][1]}")
    print(f"  Confident Accuracy  : {conf_acc*100:.1f}%")
    print(f"  Confident Prec (UP) : {conf_prec_up*100:.1f}%")
    print(f"  Confident Prec (DN) : {conf_prec_down*100:.1f}%")
    print(f"  Coverage            : {cov*100:.1f}%")

    return {
        'raw_accuracy': raw_acc,
        'raw_precision': raw_prec,
        'raw_f1': raw_f1,
        'conf_accuracy': conf_acc,
        'conf_precision_up': conf_prec_up,
        'conf_precision_down': conf_prec_down,
        'coverage': cov,
        'cm': cm.tolist()
    }

def save_model(model, feature_cols, thresh):
    package = {
        'model': model,
        'feature_cols': feature_cols,
        'confidence_high': round(float(thresh), 2),
        'confidence_low': round(float(1.0 - thresh), 2),
        'lookahead_candles': 2,
        'timeframe_minutes': 10
    }
    joblib.dump(package, MODEL_PATH)
    print(f"\n  Model Package saved to: {MODEL_PATH}")
    print(f"  Metadata: Lookahead={package['lookahead_candles']} candles ({package['timeframe_minutes']} min), Thresholds: UP>={package['confidence_high']}, DOWN<={package['confidence_low']}")

if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = load_and_split()
    run_walk_forward(X_train, y_train)
    model, thresh = train_and_tune(X_train, y_train, X_val, y_val)
    val_res = evaluate(model, X_val, y_val, thresh, label="Validation Set")
    test_res = evaluate(model, X_test, y_test, thresh, label="Held-Out Test Set (Single Pass)")
    save_model(model, feature_cols, thresh)
